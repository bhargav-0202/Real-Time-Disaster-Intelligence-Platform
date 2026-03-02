# Databricks notebook source
# Cmd 1 — Install dependencies
# %pip install openai==1.14.0

# COMMAND ----------

# dbutils.library.restartPython()

# COMMAND ----------

# Cmd 2 — 

AZURE_OPENAI_ENDPOINT   = "https://bharg-mm5xvrcj-eastus2.cognitiveservices.azure.com/"
AZURE_OPENAI_API_KEY    = "Ymkk/vB4QV2q+9liS*********Secret-Key*******vmgG7tt+AStHzKBm=="      
AZURE_OPENAI_DEPLOYMENT = "gpt-4o"
AZURE_OPENAI_VERSION    = "2025-01-01-preview"  

# Tables
SILVER_TABLE = "disaster_intelligence.silver.earthquakes"
GOLD_TABLE   = "disaster_intelligence.gold.insights"
ML_TABLE     = "disaster_intelligence.gold.ml_predictions"

print(" Config set")
print(f"   Endpoint   : {AZURE_OPENAI_ENDPOINT}")
print(f"   Deployment : {AZURE_OPENAI_DEPLOYMENT}")
print(f"   API Version: {AZURE_OPENAI_VERSION}")


# COMMAND ----------

# Cmd 3 — Test Azure OpenAI connection
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint = AZURE_OPENAI_ENDPOINT,
    api_key        = AZURE_OPENAI_API_KEY,
    api_version    = AZURE_OPENAI_VERSION
)

response = client.chat.completions.create(
    model      = AZURE_OPENAI_DEPLOYMENT,
    messages   = [
        {"role": "system", "content": "You are a disaster intelligence assistant."},
        {"role": "user",   "content": "Say 'Azure OpenAI connected successfully!' and nothing else."}
    ],
    max_tokens = 20
)

print(response.choices[0].message.content)

# COMMAND ----------

# Cmd 4 — Load latest data from Gold + ML tables
import json

# Gold insights
gold_data = spark.sql(f"""
    SELECT
        event_hour, severity, region,
        total_events, avg_magnitude, max_magnitude,
        tsunami_count, most_notable_place
    FROM {GOLD_TABLE}
    ORDER BY max_magnitude DESC
""").toPandas()

# ML predictions
ml_data = spark.sql(f"""
    SELECT
        event_id, place, magnitude, severity,
        predicted_severity, prediction_confidence,
        is_anomaly, anomaly_reason, region,
        event_time_utc, depth_km, tsunami_flag
    FROM {ML_TABLE}
    ORDER BY magnitude DESC
""").toPandas()

print(f" Data loaded")
print(f"   Gold rows : {len(gold_data)}")
print(f"   ML rows   : {len(ml_data)}")
print(f"\n📊 Top 5 earthquakes:")
print(ml_data[["place", "magnitude", "severity", "region"]].head())

# COMMAND ----------

# Cmd 5 — Feature 1: Daily Executive Summary

def generate_daily_summary(gold_df, ml_df):
    """
    Reads Gold + ML data and generates a 
    human-readable executive summary using GPT-4o
    """
    # Prepare context for the LLM
    total_events    = int(ml_df.shape[0])
    max_magnitude   = float(ml_df["magnitude"].max())
    avg_magnitude   = round(float(ml_df["magnitude"].mean()), 2)
    tsunami_count   = int(ml_df["tsunami_flag"].sum())
    anomaly_count   = int(ml_df["is_anomaly"].sum())
    top_regions     = ml_df["region"].value_counts().head(3).to_dict()
    top_event       = ml_df.iloc[0]
    critical_events = ml_df[ml_df["severity"].isin(["MAJOR", "STRONG", "CATASTROPHIC"])]

    context = f"""
    EARTHQUAKE INTELLIGENCE REPORT DATA:
    - Total earthquakes detected     : {total_events}
    - Maximum magnitude              : {max_magnitude}
    - Average magnitude              : {avg_magnitude}
    - Tsunami warnings issued        : {tsunami_count}
    - Anomalous events detected      : {anomaly_count}
    - Most active regions            : {top_regions}
    - Highest magnitude event        : M{top_event['magnitude']} at {top_event['place']}
    - Critical/Major events          : {len(critical_events)}
    
    TOP 5 EVENTS:
    {ml_df[['place','magnitude','severity','region','depth_km']].head().to_string(index=False)}
    
    ANOMALOUS EVENTS:
    {ml_df[ml_df['is_anomaly']==1][['place','magnitude','anomaly_reason']].to_string(index=False) 
     if anomaly_count > 0 else 'None detected'}
    """

    response = client.chat.completions.create(
        model    = AZURE_OPENAI_DEPLOYMENT,
        messages = [
            {
                "role": "system",
                "content": """You are a senior disaster intelligence analyst. 
                Write clear, professional executive summaries of seismic activity.
                Be concise but thorough. Use bullet points for key findings.
                Always end with a risk assessment level: LOW / MEDIUM / HIGH / CRITICAL."""
            },
            {
                "role": "user",
                "content": f"""Generate a professional daily earthquake intelligence 
                summary based on this data:\n{context}"""
            }
        ],
        max_tokens  = 600,
        temperature = 0.3
    )
    return response.choices[0].message.content

print("Generating daily summary...")
print("=" * 60)
summary = generate_daily_summary(gold_data, ml_data)
print(summary)
print("=" * 60)
print(" Daily summary generated!")

# COMMAND ----------

# Cmd 6 — Feature 2: Incident Cluster Explanation

def explain_clusters(ml_df):
    """
    Groups earthquakes by region and explains 
    what the cluster pattern means geologically
    """
    cluster_summary = ml_df.groupby("region").agg(
        count        = ("event_id",    "count"),
        avg_mag      = ("magnitude",   "mean"),
        max_mag      = ("magnitude",   "max"),
        anomalies    = ("is_anomaly",  "sum")
    ).round(2).reset_index()

    context = f"""
    EARTHQUAKE CLUSTERS BY REGION:
    {cluster_summary.to_string(index=False)}
    
    INDIVIDUAL EVENTS:
    {ml_df[['place','magnitude','depth_km','region','anomaly_reason']].to_string(index=False)}
    """

    response = client.chat.completions.create(
        model    = AZURE_OPENAI_DEPLOYMENT,
        messages = [
            {
                "role": "system",
                "content": """You are a seismologist and disaster analyst. 
                Explain earthquake cluster patterns in plain English.
                Mention possible geological causes, fault lines, or tectonic activity.
                Keep explanations clear enough for non-scientists."""
            },
            {
                "role": "user",
                "content": f"Explain these earthquake clusters and what they indicate:\n{context}"
            }
        ],
        max_tokens  = 500,
        temperature = 0.4
    )
    return response.choices[0].message.content

print("Analyzing earthquake clusters...")
print("=" * 60)
cluster_explanation = explain_clusters(ml_data)
print(cluster_explanation)
print("=" * 60)
print(" Cluster explanation generated!")

# COMMAND ----------

# Cmd 7 — Feature 3: Natural Language Q&A Interface

def earthquake_qa(user_question, ml_df, gold_df):
    """
    Takes a natural language question,
    queries the data, and returns an LLM explanation
    """
    # Give LLM the full dataset as context
    data_context = f"""
    AVAILABLE EARTHQUAKE DATA:
    
    Individual Events (Silver/ML layer):
    {ml_df[['place','magnitude','severity','depth_km',
            'region','tsunami_flag','is_anomaly',
            'event_time_utc','predicted_severity',
            'prediction_confidence']].to_string(index=False)}
    
    Aggregated Insights (Gold layer):
    {gold_df.to_string(index=False)}
    """

    response = client.chat.completions.create(
        model    = AZURE_OPENAI_DEPLOYMENT,
        messages = [
            {
                "role": "system",
                "content": """You are an expert disaster intelligence analyst with access 
                to real-time earthquake data. Answer questions accurately using ONLY 
                the data provided. Be specific — mention actual place names, magnitudes, 
                and times from the data. If the data doesn't contain enough information 
                to answer, say so clearly."""
            },
            {
                "role": "user",
                "content": f"""
                DATA:\n{data_context}
                
                QUESTION: {user_question}
                
                Provide a detailed, accurate answer based strictly on the data above.
                """
            }
        ],
        max_tokens  = 400,
        temperature = 0.2
    )
    return response.choices[0].message.content


# Test with sample questions
questions = [
    "What were the most critical earthquakes in the last 24 hours and why?",
    "Which region had the most seismic activity and what does that suggest?",
    "Were there any anomalous earthquakes detected? Explain what made them unusual.",
]

for q in questions:
    print(f"\n QUESTION: {q}")
    print("-" * 60)
    answer = earthquake_qa(q, ml_data, gold_data)
    print(answer)
    print("=" * 60)

print("\n Q&A interface working!")

# COMMAND ----------

# Cmd 8 — Feature 4: Auto Response Playbook Generator

def generate_playbook(ml_df):
    """
    Automatically generates emergency response playbooks
    for the most significant earthquake detected
    """
    # Focus on highest magnitude event
    top_event = ml_df.iloc[0]

    context = f"""
    EARTHQUAKE EVENT DETAILS:
    - Location      : {top_event['place']}
    - Magnitude     : M{top_event['magnitude']}
    - Severity      : {top_event['severity']}
    - Depth         : {top_event['depth_km']} km
    - Tsunami flag  : {top_event['tsunami_flag']}
    - Region        : {top_event['region']}
    - Anomaly       : {top_event['anomaly_reason']}
    - ML Prediction : {top_event['predicted_severity']} 
                      (confidence: {round(top_event['prediction_confidence']*100, 1)}%)
    """

    response = client.chat.completions.create(
        model    = AZURE_OPENAI_DEPLOYMENT,
        messages = [
            {
                "role": "system",
                "content": """You are an emergency response coordinator specializing 
                in earthquake disaster management. Generate detailed, actionable 
                response playbooks. Structure them with clear sections:
                1. Immediate Actions (0-1 hour)
                2. Short-term Response (1-24 hours)  
                3. Monitoring Requirements
                4. Communication Plan
                5. Resource Deployment
                Be specific and practical."""
            },
            {
                "role": "user",
                "content": f"""Generate a complete emergency response playbook 
                for this earthquake event:\n{context}"""
            }
        ],
        max_tokens  = 700,
        temperature = 0.2
    )
    return response.choices[0].message.content

print("Generating emergency response playbook...")
print("=" * 60)
playbook = generate_playbook(ml_data)
print(playbook)
print("=" * 60)
print(" Response playbook generated!")

# COMMAND ----------

# Cmd 9 — Save all GenAI outputs to Delta table for history

from pyspark.sql import Row
import datetime

outputs = [
    Row(
        output_type  = "daily_summary",
        content      = summary,
        generated_at = datetime.datetime.utcnow().isoformat(),
        model_used   = AZURE_OPENAI_DEPLOYMENT
    ),
    Row(
        output_type  = "cluster_explanation",
        content      = cluster_explanation,
        generated_at = datetime.datetime.utcnow().isoformat(),
        model_used   = AZURE_OPENAI_DEPLOYMENT
    ),
    Row(
        output_type  = "response_playbook",
        content      = playbook,
        generated_at = datetime.datetime.utcnow().isoformat(),
        model_used   = AZURE_OPENAI_DEPLOYMENT
    ),
]

genai_df = spark.createDataFrame(outputs)

(
    genai_df.write
    .format("delta")
    .mode("append")
    .saveAsTable("disaster_intelligence.gold.genai_outputs")
)

print("   GenAI outputs saved to Delta table!")
print("   Table: disaster_intelligence.gold.genai_outputs")

# COMMAND ----------

# Cmd 10 — View saved outputs
spark.sql("""
    SELECT output_type, generated_at, 
           LEFT(content, 200) AS content_preview
    FROM disaster_intelligence.gold.genai_outputs
    ORDER BY generated_at DESC
""").show(truncate=False)

# COMMAND ----------

# Checking  tables 
spark.sql("SHOW TABLES IN disaster_intelligence.bronze").show()
spark.sql("SHOW TABLES IN disaster_intelligence.silver").show()
spark.sql("SHOW TABLES IN disaster_intelligence.gold").show()

# COMMAND ----------

tables = [
    "disaster_intelligence.bronze.earthquakes",
    "disaster_intelligence.silver.earthquakes",
    "disaster_intelligence.gold.insights",
    "disaster_intelligence.gold.ml_predictions",
    "disaster_intelligence.gold.genai_outputs"
]

print("=" * 50)
print("TABLE STATUS CHECK")
print("=" * 50)

for table in tables:
    try:
        count = spark.sql(f"SELECT COUNT(*) AS cnt FROM {table}").collect()[0][0]
        print(f" {table}")
        print(f"   Rows: {count}")
    except Exception as e:
        print(f" {table}")
        print(f"   Error: {str(e)[:80]}")
    print()

# COMMAND ----------

