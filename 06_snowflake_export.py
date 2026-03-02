# Databricks notebook source
# Cmd 1 — Install Snowflake connector
%pip install snowflake-connector-python snowflake-sqlalchemy

# COMMAND ----------

# Cmd 2 — Snowflake credentials
SNOWFLAKE_ACCOUNT   = "LIXXXXXX"
SNOWFLAKE_USER      = "BHARGAV"
SNOWFLAKE_PASSWORD  = "Password"   
SNOWFLAKE_DATABASE  = "DISASTER_INTELLIGENCE"
SNOWFLAKE_SCHEMA    = "PUBLIC"
SNOWFLAKE_WAREHOUSE = "COMPUTE_WH"
SNOWFLAKE_ROLE      = "ACCOUNTADMIN"

print("  Snowflake config set")
print(f"   Account   : {SNOWFLAKE_ACCOUNT}")
print(f"   User      : {SNOWFLAKE_USER}")
print(f"   Database  : {SNOWFLAKE_DATABASE}")
print(f"   Warehouse : {SNOWFLAKE_WAREHOUSE}")

# COMMAND ----------

# Cmd 3 — Test Snowflake connection
import snowflake.connector

try:
    conn = snowflake.connector.connect(
        account   = SNOWFLAKE_ACCOUNT,
        user      = SNOWFLAKE_USER,
        password  = SNOWFLAKE_PASSWORD,
        warehouse = SNOWFLAKE_WAREHOUSE,
        database  = SNOWFLAKE_DATABASE,
        schema    = SNOWFLAKE_SCHEMA,
        role      = SNOWFLAKE_ROLE
    )
    cursor = conn.cursor()
    cursor.execute("SELECT CURRENT_ACCOUNT(), CURRENT_USER(), CURRENT_WAREHOUSE()")
    row = cursor.fetchone()
    print(f" Snowflake connected successfully!")
    print(f"   Account   : {row[0]}")
    print(f"   User      : {row[1]}")
    print(f"   Warehouse : {row[2]}")
    conn.close()
except Exception as e:
    print(f" Connection failed: {e}")

# COMMAND ----------

# Cmd 4 — Export Gold Insights to Snowflake
import snowflake.connector
import pandas as pd

def export_to_snowflake(spark_table, sf_table, mode="overwrite"):
    """
    Reads a Spark Delta table and writes it to Snowflake
    """
    print(f"Exporting {spark_table} → Snowflake:{sf_table}...")
    
    # Read from Delta
    df = spark.table(spark_table)
    count = df.count()
    
    # Convert to Pandas
    pdf = df.toPandas()
    
    # Clean column names — Snowflake prefers uppercase
    pdf.columns = [c.upper() for c in pdf.columns]
    
    # Connect to Snowflake
    conn = snowflake.connector.connect(
        account   = SNOWFLAKE_ACCOUNT,
        user      = SNOWFLAKE_USER,
        password  = SNOWFLAKE_PASSWORD,
        warehouse = SNOWFLAKE_WAREHOUSE,
        database  = SNOWFLAKE_DATABASE,
        schema    = SNOWFLAKE_SCHEMA,
        role      = SNOWFLAKE_ROLE
    )
    
    # Write using pandas + snowflake connector
    from snowflake.connector.pandas_tools import write_pandas
    
    success, nchunks, nrows, _ = write_pandas(
        conn            = conn,
        df              = pdf,
        table_name      = sf_table,
        auto_create_table = True,
        overwrite       = (mode == "overwrite")
    )
    
    conn.close()
    
    if success:
        print(f" Exported successfully!")
        print(f"   Rows exported : {nrows}")
        print(f"   Chunks        : {nchunks}")
    else:
        print(f" Export failed!")
    
    return nrows

# Export Gold Insights
export_to_snowflake(
    "disaster_intelligence.gold.insights",
    "GOLD_INSIGHTS"
)

# COMMAND ----------

# Cmd 5 — Export ML Predictions
export_to_snowflake(
    "disaster_intelligence.gold.ml_predictions",
    "ML_PREDICTIONS"
)

# COMMAND ----------

# Cmd 6 — Export GenAI Outputs
export_to_snowflake(
    "disaster_intelligence.gold.genai_outputs",
    "GENAI_OUTPUTS"
)

# COMMAND ----------

# Cmd 7 — Verify all tables in Snowflake
conn = snowflake.connector.connect(
    account   = SNOWFLAKE_ACCOUNT,
    user      = SNOWFLAKE_USER,
    password  = SNOWFLAKE_PASSWORD,
    warehouse = SNOWFLAKE_WAREHOUSE,
    database  = SNOWFLAKE_DATABASE,
    schema    = SNOWFLAKE_SCHEMA,
    role      = SNOWFLAKE_ROLE
)
cursor = conn.cursor()
cursor.execute("SHOW TABLES IN DISASTER_INTELLIGENCE.PUBLIC")
tables = cursor.fetchall()

print("=" * 50)
print("SNOWFLAKE TABLES")
print("=" * 50)
for t in tables:
    print(f" {t[1]:30} rows: {t[2]}")

conn.close()

# COMMAND ----------

# Cmd 8 — Quick validation query on Snowflake
conn = snowflake.connector.connect(
    account   = SNOWFLAKE_ACCOUNT,
    user      = SNOWFLAKE_USER,
    password  = SNOWFLAKE_PASSWORD,
    warehouse = SNOWFLAKE_WAREHOUSE,
    database  = SNOWFLAKE_DATABASE,
    schema    = SNOWFLAKE_SCHEMA,
    role      = SNOWFLAKE_ROLE
)
cursor = conn.cursor()

cursor.execute("""
    SELECT 
        SEVERITY,
        REGION,
        COUNT(*) AS EVENTS,
        MAX(MAX_MAGNITUDE) AS MAX_MAG
    FROM GOLD_INSIGHTS
    GROUP BY SEVERITY, REGION
    ORDER BY MAX_MAG DESC
    LIMIT 10
""")

results = cursor.fetchall()
print("=" * 60)
print("TOP EARTHQUAKE INSIGHTS IN SNOWFLAKE")
print("=" * 60)
print(f"{'SEVERITY':<15} {'REGION':<20} {'EVENTS':<8} {'MAX MAG'}")
print("-" * 60)
for row in results:
    print(f"{str(row[0]):<15} {str(row[1]):<20} {str(row[2]):<8} {row[3]}")

conn.close()

# COMMAND ----------

