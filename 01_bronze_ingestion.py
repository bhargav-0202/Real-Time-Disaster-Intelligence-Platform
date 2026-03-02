# Databricks notebook source
# MAGIC %md
# MAGIC ## Real-Time Disaster Intelligence
# MAGIC ### Notebook: 01 — Bronze Ingestion (USGS via Event Hubs → Bronze Delta)

# COMMAND ----------

# Cmd 1 — Install Python dependencies on the cluster
# (Run once; skip on subsequent runs if already installed)

# %pip install azure-eventhub requests

# COMMAND ----------

# Cmd 2 — Credentials
AZURE_STORAGE_ACCOUNT_NAME = "disasterintelstore"  
AZURE_STORAGE_ACCESS_KEY   = "Ymkk/vB4QV2q+9liS*********Secret-Key*******vmgG7tt+AStHzKBm=="       

# ADLS paths
LANDING_PATH     = f"abfss://earthquake-raw@{AZURE_STORAGE_ACCOUNT_NAME}.dfs.core.windows.net/landing/"
CHECKPOINT_PATH  = f"abfss://earthquake-raw@{AZURE_STORAGE_ACCOUNT_NAME}.dfs.core.windows.net/checkpoints/bronze/"
BRONZE_TABLE     = "disaster_intelligence.bronze.earthquakes"

print("Config set")
print(f"   Landing path : {LANDING_PATH}")

# COMMAND ----------

# Cmd 3 — Configure ADLS Gen2 access for checkpoint storage

# Cmd 3 — Configure storage access
spark.conf.set(
    f"fs.azure.account.key.{AZURE_STORAGE_ACCOUNT_NAME}.dfs.core.windows.net",
    AZURE_STORAGE_ACCESS_KEY
)
print(" Storage access configured")

# COMMAND ----------

# Cmd 4 — Create storage container and folders
dbutils.fs.mkdirs(LANDING_PATH)
dbutils.fs.mkdirs(CHECKPOINT_PATH)
print(" Storage folders created")
print(f"   {LANDING_PATH}")
print(f"   {CHECKPOINT_PATH}")

# COMMAND ----------

# Cmd 5 — Define JSON schema (mirrors what usgs_producer.py sends)

# Cmd 5 — USGS Fetcher: fetch live data and write to ADLS as JSON files
import requests
import json
import datetime

def fetch_and_store_usgs():
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson"
    
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    data = response.json()
    features = data.get("features", [])
    
    print(f"Fetched {len(features)} earthquakes from USGS")
    
    events = []
    for f in features:
        props  = f.get("properties", {})
        coords = f.get("geometry", {}).get("coordinates", [None, None, None])
        
        event = {
            "event_id"               : f.get("id"),
            "source"                 : "USGS",
            "magnitude"              : props.get("mag"),
            "magnitude_type"         : props.get("magType"),
            "depth_km"               : coords[2],
            "latitude"               : coords[1],
            "longitude"              : coords[0],
            "place"                  : props.get("place"),
            "event_type"             : props.get("type"),
            "event_time_epoch_ms"    : props.get("time"),
            "event_time_utc"         : datetime.datetime.utcfromtimestamp(
                                           props["time"] / 1000
                                       ).isoformat() + "Z" if props.get("time") else None,
            "alert_level"            : props.get("alert"),
            "tsunami_flag"           : bool(props.get("tsunami")),
            "significance"           : props.get("sig"),
            "felt_reports"           : props.get("felt"),
            "cdi"                    : props.get("cdi"),
            "mmi"                    : props.get("mmi"),
            "status"                 : props.get("status"),
            "network"                : props.get("net"),
            "gap_deg"                : props.get("gap"),
            "rms"                    : props.get("rms"),
            "nst"                    : props.get("nst"),
            "detail_url"             : props.get("detail"),
            "event_url"              : props.get("url"),
            "ingestion_timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        }
        events.append(event)
    
    # Write as a single JSON file to ADLS landing zone
    timestamp    = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_path    = f"{LANDING_PATH}earthquakes_{timestamp}.json"
    json_content = "\n".join(json.dumps(e) for e in events)  # one JSON per line
    
    dbutils.fs.put(file_path, json_content, overwrite=True)
    print(f" Written {len(events)} events → {file_path}")
    return len(events)

# Run it now — first live test!
count = fetch_and_store_usgs()
print(f"\n Total events stored: {count}")

# COMMAND ----------

# Cmd 6 — Verify the file landed in storage
files = dbutils.fs.ls(LANDING_PATH)
print(f"Files in landing zone: {len(files)}")
for f in files:
    print(f"   {f.name}  ({f.size} bytes)")

# COMMAND ----------

# Cmd 7 — Define schema
from pyspark.sql.types import *

EARTHQUAKE_SCHEMA = StructType([
    StructField("event_id",                StringType(),  True),
    StructField("source",                  StringType(),  True),
    StructField("magnitude",               DoubleType(),  True),
    StructField("magnitude_type",          StringType(),  True),
    StructField("depth_km",               DoubleType(),  True),
    StructField("latitude",               DoubleType(),  True),
    StructField("longitude",              DoubleType(),  True),
    StructField("place",                  StringType(),  True),
    StructField("event_type",             StringType(),  True),
    StructField("event_time_epoch_ms",    LongType(),    True),
    StructField("event_time_utc",         StringType(),  True),
    StructField("alert_level",            StringType(),  True),
    StructField("tsunami_flag",           BooleanType(), True),
    StructField("significance",           IntegerType(), True),
    StructField("felt_reports",           IntegerType(), True),
    StructField("cdi",                    DoubleType(),  True),
    StructField("mmi",                    DoubleType(),  True),
    StructField("status",                 StringType(),  True),
    StructField("network",                StringType(),  True),
    StructField("gap_deg",                DoubleType(),  True),
    StructField("rms",                    DoubleType(),  True),
    StructField("nst",                    IntegerType(), True),
    StructField("detail_url",             StringType(),  True),
    StructField("event_url",              StringType(),  True),
    StructField("ingestion_timestamp_utc",StringType(),  True),
])

print("Schema defined")

# COMMAND ----------

# Cmd 8 — Auto Loader streaming: landing zone → Bronze Delta table
from pyspark.sql.functions import current_timestamp, input_file_name

bronze_stream = (
    spark.readStream
    .format("cloudFiles")                        # Auto Loader
    .option("cloudFiles.format", "json")
    .option("cloudFiles.schemaLocation", f"{CHECKPOINT_PATH}schema/")
    .schema(EARTHQUAKE_SCHEMA)
    .load(LANDING_PATH)
    .withColumn("bronze_load_timestamp", current_timestamp())
    .withColumn("source_file",           input_file_name())
)

query = (
    bronze_stream.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", CHECKPOINT_PATH)
    .option("mergeSchema", "true")
    .trigger(processingTime="30 seconds")
    .toTable(BRONZE_TABLE)
)

print(f" Auto Loader stream running!")
print(f"   Reading from : {LANDING_PATH}")
print(f"   Writing to   : {BRONZE_TABLE}")

# COMMAND ----------

# Cmd 9 — Continuous USGS fetcher (runs every 60 seconds automatically)
import time
import requests
import json
import datetime

def fetch_and_store_usgs():
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    features = response.json().get("features", [])

    events = []
    for f in features:
        props  = f.get("properties", {})
        coords = f.get("geometry", {}).get("coordinates", [None, None, None])
        events.append({
            "event_id"               : f.get("id"),
            "source"                 : "USGS",
            "magnitude"              : props.get("mag"),
            "magnitude_type"         : props.get("magType"),
            "depth_km"               : coords[2],
            "latitude"               : coords[1],
            "longitude"              : coords[0],
            "place"                  : props.get("place"),
            "event_type"             : props.get("type"),
            "event_time_epoch_ms"    : props.get("time"),
            "event_time_utc"         : datetime.datetime.utcfromtimestamp(
                                           props["time"] / 1000
                                       ).isoformat() + "Z" if props.get("time") else None,
            "alert_level"            : props.get("alert"),
            "tsunami_flag"           : bool(props.get("tsunami")),
            "significance"           : props.get("sig"),
            "felt_reports"           : props.get("felt"),
            "cdi"                    : props.get("cdi"),
            "mmi"                    : props.get("mmi"),
            "status"                 : props.get("status"),
            "network"                : props.get("net"),
            "gap_deg"                : props.get("gap"),
            "rms"                    : props.get("rms"),
            "nst"                    : props.get("nst"),
            "detail_url"             : props.get("detail"),
            "event_url"              : props.get("url"),
            "ingestion_timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        })

    timestamp    = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_path    = f"{LANDING_PATH}earthquakes_{timestamp}.json"
    json_content = "\n".join(json.dumps(e) for e in events)
    dbutils.fs.put(file_path, json_content, overwrite=True)
    return len(events)

# ─── Continuous loop ──────
POLL_INTERVAL = 60   
MAX_POLLS     = 60   

poll = 0
while MAX_POLLS is None or poll < MAX_POLLS:
    poll += 1
    now   = datetime.datetime.utcnow().strftime("%H:%M:%S")
    count = fetch_and_store_usgs()
    print(f"[{now}] Poll #{poll} → {count} events written to landing zone")
    time.sleep(POLL_INTERVAL)



# COMMAND ----------

# query.stop()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     *
# MAGIC FROM disaster_intelligence.bronze.earthquakes

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC     COUNT(*)  AS total_events,
# MAGIC     COUNT(DISTINCT event_id) AS unique_events,
# MAGIC     ROUND(AVG(magnitude), 2) AS avg_magnitude,
# MAGIC     MAX(magnitude) AS max_magnitude,
# MAGIC     SUM(CASE WHEN tsunami_flag = TRUE THEN 1 ELSE 0 END) AS tsunami_events,
# MAGIC     MIN(event_time_utc) AS earliest_event,
# MAGIC     MAX(event_time_utc) AS latest_event
# MAGIC FROM disaster_intelligence.bronze.earthquakes

# COMMAND ----------


spark.sql("""
SELECT
    event_id,
    event_time_utc,
    place,
    magnitude,
    severity,
    depth_category,
    region,
    alert_level,
    tsunami_flag
FROM disaster_intelligence.silver.earthquakes
ORDER BY magnitude DESC
""").show(11, truncate=False)

# COMMAND ----------

spark.sql("""
SELECT
    event_hour,
    severity,
    region,
    total_events,
    avg_magnitude,
    max_magnitude,
    tsunami_count,
    most_notable_place
FROM disaster_intelligence.gold.insights
ORDER BY max_magnitude DESC
""").show(9, truncate=False)

# COMMAND ----------

spark.streams.active

# COMMAND ----------

