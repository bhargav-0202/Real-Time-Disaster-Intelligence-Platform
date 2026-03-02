# Databricks notebook source
# Cmd 1 — Credentials (same as before)

AZURE_STORAGE_ACCOUNT_NAME = "disasterintelstore"  
AZURE_STORAGE_ACCESS_KEY   = "Ymkk/vB4QV2q+9liS*********Secret-Key*******vmgG7tt+AS=="        

# Paths
SILVER_CHECKPOINT = (
    f"abfss://earthquake-raw@{AZURE_STORAGE_ACCOUNT_NAME}"
    f".dfs.core.windows.net/checkpoints/silver/"
)
GOLD_CHECKPOINT = (
    f"abfss://earthquake-raw@{AZURE_STORAGE_ACCOUNT_NAME}"
    f".dfs.core.windows.net/checkpoints/gold/"
)

# Table names
BRONZE_TABLE = "disaster_intelligence.bronze.earthquakes"
SILVER_TABLE = "disaster_intelligence.silver.earthquakes"
GOLD_TABLE   = "disaster_intelligence.gold.insights"


# COMMAND ----------

# Cmd 2 — Configure storage access
spark.conf.set(
    f"fs.azure.account.key.{AZURE_STORAGE_ACCOUNT_NAME}.dfs.core.windows.net",
    AZURE_STORAGE_ACCESS_KEY
)


# COMMAND ----------

# Cmd 3 — Create Silver and Gold schemas
spark.sql("CREATE SCHEMA IF NOT EXISTS disaster_intelligence.silver")

spark.sql("CREATE SCHEMA IF NOT EXISTS disaster_intelligence.gold")


# COMMAND ----------

# Cmd 4 — Build Silver table
# Deduplicates, cleans, adds severity label and broad region

from pyspark.sql.functions import (
    col, to_timestamp, when, udf, row_number, current_timestamp
)
from pyspark.sql.types import StringType
from pyspark.sql.window import Window

# Read Bronze as a batch (we'll add streaming in a moment)
bronze_df = spark.table(BRONZE_TABLE)

# Step 1 — Deduplicate: keep the latest version of each event_id
window = Window.partitionBy("event_id").orderBy(col("bronze_load_timestamp").desc())

deduped_df = (
    bronze_df
    .filter(col("event_id").isNotNull())
    .withColumn("row_num", row_number().over(window))
    .filter(col("row_num") == 1)
    .drop("row_num")
)

# Step 2 — Add severity label based on magnitude
silver_df = (
    deduped_df

    # Cast event time to proper timestamp
    .withColumn(
        "event_timestamp",
        to_timestamp(col("event_time_utc"), "yyyy-MM-dd'T'HH:mm:ss.SSSSSS'Z'")
    )

    # Severity classification
    .withColumn(
        "severity",
        when(col("magnitude") >= 7.0, "CATASTROPHIC")
        .when(col("magnitude") >= 6.0, "MAJOR")
        .when(col("magnitude") >= 5.0, "STRONG")
        .when(col("magnitude") >= 4.0, "MODERATE")
        .when(col("magnitude") >= 3.0, "MINOR")
        .when(col("magnitude") >= 2.0, "MICRO")
        .otherwise("NEGLIGIBLE")
    )

    # Depth classification
    .withColumn(
        "depth_category",
        when(col("depth_km") <= 70,  "SHALLOW")
        .when(col("depth_km") <= 300, "INTERMEDIATE")
        .otherwise("DEEP")
    )

    # Alert level — fill nulls
    .withColumn(
        "alert_level",
        when(col("alert_level").isNull(), "none")
        .otherwise(col("alert_level"))
    )

    # Broad region from coordinates
    .withColumn(
        "region",
        when((col("latitude") >= 24)  & (col("latitude") <= 49)  &
             (col("longitude") >= -125) & (col("longitude") <= -66),  "North America")
        .when((col("latitude") >= 35)  & (col("latitude") <= 72)  &
             (col("longitude") >= -10)  & (col("longitude") <= 40),   "Europe")
        .when((col("latitude") >= -10) & (col("latitude") <= 55)  &
             (col("longitude") >= 60)   & (col("longitude") <= 150),  "Asia")
        .when((col("latitude") >= -55) & (col("latitude") <= 15)  &
             (col("longitude") >= -82)  & (col("longitude") <= -34),  "South America")
        .when((col("latitude") >= -35) & (col("latitude") <= 37)  &
             (col("longitude") >= -18)  & (col("longitude") <= 52),   "Africa")
        .when((col("latitude") >= -50) & (col("latitude") <= -10) &
             (col("longitude") >= 110)  & (col("longitude") <= 180),  "Oceania")
        .when((col("latitude") >= 51)  & (col("latitude") <= 72)  &
             (col("longitude") >= -168) & (col("longitude") <= -130), "Alaska/Arctic")
        .otherwise("Pacific/Other")
    )

    # Silver load timestamp
    .withColumn("silver_load_timestamp", current_timestamp())

    # Select final columns
    .select(
        "event_id", "source", "event_timestamp", "event_time_utc",
        "magnitude", "magnitude_type", "severity",
        "depth_km", "depth_category",
        "latitude", "longitude", "place", "region",
        "alert_level", "tsunami_flag", "significance",
        "felt_reports", "cdi", "mmi",
        "status", "network", "gap_deg", "rms", "nst",
        "detail_url", "event_url",
        "ingestion_timestamp_utc", "silver_load_timestamp"
    )
)


# COMMAND ----------

# Cmd 5 — Write Silver table
(
    silver_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SILVER_TABLE)
)


# COMMAND ----------

# Cmd 6 — Validate Silver
# %sql
# SELECT
#     severity,
#     region,
#     COUNT(*)              AS event_count,
#     ROUND(AVG(magnitude), 2) AS avg_magnitude,
#     MAX(magnitude)        AS max_magnitude,
#     SUM(CASE WHEN tsunami_flag = TRUE THEN 1 ELSE 0 END) AS tsunamis
# FROM disaster_intelligence.silver.earthquakes
# GROUP BY severity, region
# ORDER BY max_magnitude DESC

# COMMAND ----------

# Cmd 7 — Build Gold aggregations
from pyspark.sql.functions import (
    count, avg, max, min, sum, round,
    date_trunc, countDistinct, first
)

silver = spark.table(SILVER_TABLE)

gold_df = (
    silver
    # Aggregate by hour + severity + region
    .withColumn("event_hour", date_trunc("hour", col("event_timestamp")))
    .groupBy("event_hour", "severity", "region")
    .agg(
        count("*")                          .alias("total_events"),
        countDistinct("event_id")           .alias("unique_events"),
        round(avg("magnitude"), 2)          .alias("avg_magnitude"),
        max("magnitude")                    .alias("max_magnitude"),
        min("magnitude")                    .alias("min_magnitude"),
        round(avg("depth_km"), 2)           .alias("avg_depth_km"),
        sum(when(col("tsunami_flag") == True, 1).otherwise(0)).alias("tsunami_count"),
        sum(when(col("alert_level") != "none", 1).otherwise(0)).alias("alerted_events"),
        max("significance")                 .alias("max_significance"),
        first("place")                      .alias("most_notable_place"),
        current_timestamp()                 .alias("gold_load_timestamp")
    )
    .orderBy(col("event_hour").desc(), col("max_magnitude").desc())
)

print(f" Gold aggregation complete")


# COMMAND ----------

# Cmd 8 — Write Gold table
(
    gold_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(GOLD_TABLE)
)

print(f"Gold table written: {GOLD_TABLE}")

# COMMAND ----------

# Cmd 9 — Final validation across all 3 layers
spark.sql("""
SELECT 'BRONZE' AS layer, COUNT(*) AS rows FROM disaster_intelligence.bronze.earthquakes
UNION ALL
SELECT 'SILVER' AS layer, COUNT(*) AS rows FROM disaster_intelligence.silver.earthquakes
UNION ALL
SELECT 'GOLD'   AS layer, COUNT(*) AS rows FROM disaster_intelligence.gold.insights
""").show()


# COMMAND ----------

