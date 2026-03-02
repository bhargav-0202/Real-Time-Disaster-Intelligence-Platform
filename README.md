# Real-Time-Disaster-Intelligence-Platform  + GenAI Summarization System

> **A production-grade, end-to-end real-time data engineering and AI project built on Azure Databricks, Delta Lake, Snowflake, and Azure OpenAI GPT-4o.**

---

## 📌 Project Overview

This project ingests live earthquake data from the USGS (United States Geological Survey) API every 60 seconds, processes it through a Medallion Architecture (Bronze → Silver → Gold), applies machine learning for anomaly detection and severity classification, generates intelligent natural language summaries using GPT-4o, exports curated data to Snowflake, and visualizes insights on Power BI dashboards — all in real time.

**Built to demonstrate:** Streaming data engineering, Delta Lake architecture, MLflow model tracking, GenAI integration, and cloud-native data warehousing — skills that are in high demand across the data engineering and data science industry.

---

## 🏗️ Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA SOURCES                                 │
│   USGS Earthquake GeoJSON Feed (updates every 1 minute)         │
│   https://earthquake.usgs.gov/earthquakes/feed/v1.0/            │
└────────────────────────┬────────────────────────────────────────┘
                         │ Python Fetcher (every 60s)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AZURE DATA LAKE (ADLS Gen2)                  │
│   Container: earthquake-raw                                     │
│   Path: /landing/earthquakes_YYYYMMDD_HHMMSS.json               │
│   Format: NDJSON (one event per line)                           │
└────────────────────────┬────────────────────────────────────────┘
                         │ Databricks Auto Loader (CloudFiles)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│               DATABRICKS + DELTA LAKE (MEDALLION)               │
│                                                                 │
│  🥉 BRONZE — disaster_intelligence.bronze.earthquakes           │
│     Raw append-only data, 807+ rows, no transformations         │
│                         │                                       │
│                         ▼ Dedup + Enrich                        │
│  🥈 SILVER — disaster_intelligence.silver.earthquakes           │
│     Deduplicated, severity labels, region classification        │
│                         │                                       │
│                         ▼ Aggregate                             │
│  🥇 GOLD — disaster_intelligence.gold.insights                  │
│     Hourly aggregations by severity and region                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
┌─────────────────────┐  ┌──────────────────────────────────────┐
│    ML LAYER         │  │         GENAI LAYER                  │
│  XGBoost Classifier │  │   Azure OpenAI GPT-4o                │
│  Anomaly Detection  │  │   Daily Summaries                    │
│  MLflow Tracking    │  │   Cluster Explanations               │
│  gold.ml_predictions│  │   Natural Language Q&A               │
└─────────────────────┘  │   Emergency Playbooks                │
                         │   gold.genai_outputs                 │
                         └──────────────┬───────────────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────┐
                         │       SNOWFLAKE           │
                         │  DISASTER_INTELLIGENCE DB │
                         │  GOLD_INSIGHTS            │
                         │  ML_PREDICTIONS           │
                         │  GENAI_OUTPUTS            │
                         └──────────────┬────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────┐
                         │       POWER BI            │
                         │  Live Earthquake Dashboard│
                         │  Severity Maps            │
                         │  Anomaly Charts           │
                         └──────────────────────────┘
```

### Streaming Architecture (Auto Loader)

```
USGS API ──► Python Fetcher ──► ADLS Landing Zone
                (60s poll)      (NDJSON files)
                                       │
                                       ▼
                           Databricks Auto Loader
                           (CloudFiles format)
                           trigger: 30 seconds
                                       │
                                       ▼
                           Bronze Delta Table
                           (append-only, streaming)
```

### Medallion Architecture

```
BRONZE                    SILVER                    GOLD
───────                   ──────                    ────
Raw JSON data    ──►      Deduplicated     ──►      Aggregated
No transforms             Severity labels           Hourly stats
Append-only               Region enriched           ML-enriched
807+ rows                 31 unique events           19 insight rows
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Data Source | USGS GeoJSON Feed | Live earthquake data, free, no API key |
| Cloud Platform | Microsoft Azure | Infrastructure hosting |
| Storage | ADLS Gen2 | Raw file landing zone, Delta checkpoints |
| Processing | Azure Databricks 17.3 LTS (Spark 4.0) | Streaming + batch processing |
| Table Format | Delta Lake (Unity Catalog) | ACID transactions, time travel |
| ML Framework | XGBoost + Scikit-learn | Severity classification |
| Experiment Tracking | MLflow | Model versioning and metrics |
| GenAI | Azure OpenAI GPT-4o | Summarization, Q&A, playbooks |
| Data Warehouse | Snowflake (Enterprise, AWS) | Curated data serving layer |
| Visualization | Power BI | Executive dashboards |
| Language | Python + PySpark + SQL | Primary development languages |
| Ingestion Pattern | Databricks Auto Loader | Schema evolution, exactly-once |

---

## 📁 Project Structure

```
disaster_intelligence/
├── notebooks/
│   ├── 01_bronze_ingestion.py      # Auto Loader streaming pipeline
│   ├── 02_silver_gold.py           # Medallion transformations
│   ├── 03_ml_layer.py              # XGBoost + Anomaly Detection
│   ├── 04_genai_layer.py           # GPT-4o integration
│   └── 05_snowflake_export.py      # Snowflake data export
└── README.md                       # This file
```

---

## 🚀 How to Run Locally

### Prerequisites

| Requirement | Version / Notes |
|-------------|----------------|
| Azure Subscription | Free trial ($200 credit) or Pay-as-you-go |
| Databricks Workspace | 17.3 LTS (Spark 4.0, Scala 2.13) |
| Azure Storage Account | ADLS Gen2, create container: `earthquake-raw` |
| Azure OpenAI Resource | GPT-4o deployment, East US 2 region |
| Snowflake Account | Any edition, AWS or Azure cloud |
| Power BI Desktop | Free download from powerbi.microsoft.com |

### Step 1 — Azure Setup

```bash
# 1. Create Resource Group
az group create --name disaster-intelligence-rg --location eastus2

# 2. Create Storage Account
az storage account create \
  --name yourstorageaccount \
  --resource-group disaster-intelligence-rg \
  --sku Standard_LRS \
  --kind StorageV2 \
  --hierarchical-namespace true

# 3. Create container
az storage container create \
  --name earthquake-raw \
  --account-name yourstorageaccount
```

### Step 2 — Databricks Cluster Configuration

```
Runtime: 13.3 LTS or 17.3 LTS (Spark 4.0)
Node type: Standard_DS3_v2 (minimum)
Libraries (PyPI):
  - requests
  - azure-eventhub
  - xgboost
  - scikit-learn
  - mlflow
  - openai==1.14.0
  - snowflake-connector-python
  - snowflake-sqlalchemy
```

### Step 3 — Unity Catalog Setup

```sql
-- Run in Databricks SQL
CREATE CATALOG IF NOT EXISTS disaster_intelligence;
CREATE SCHEMA IF NOT EXISTS disaster_intelligence.bronze;
CREATE SCHEMA IF NOT EXISTS disaster_intelligence.silver;
CREATE SCHEMA IF NOT EXISTS disaster_intelligence.gold;
```

### Step 4 — Configure Credentials

In each notebook, update the credentials section:

```python
# Storage
AZURE_STORAGE_ACCOUNT_NAME = "your_storage_account"
AZURE_STORAGE_ACCESS_KEY   = "your_access_key"

# Azure OpenAI
AZURE_OPENAI_ENDPOINT   = "https://your-resource.cognitiveservices.azure.com/"
AZURE_OPENAI_API_KEY    = "your_api_key"
AZURE_OPENAI_DEPLOYMENT = "gpt-4o"
AZURE_OPENAI_VERSION    = "2025-01-01-preview"

# Snowflake
SNOWFLAKE_ACCOUNT   = "your_account_locator"
SNOWFLAKE_USER      = "your_username"
SNOWFLAKE_PASSWORD  = "your_password"
```

### Step 5 — Run Notebooks in Order

```
1. 01_bronze_ingestion.py    → Start USGS fetcher + Auto Loader stream
2. 02_silver_gold.py         → Run Silver + Gold transformations
3. 03_ml_layer.py            → Train ML model + generate predictions
4. 04_genai_layer.py         → Generate AI summaries and playbooks
5. 05_snowflake_export.py    → Export to Snowflake
```

### Step 6 — Power BI Connection

```
1. Open Power BI Desktop
2. Get Data → Snowflake
3. Server: your_locator.snowflakecomputing.com
4. Warehouse: COMPUTE_WH
5. Database: DISASTER_INTELLIGENCE
6. Import: GOLD_INSIGHTS, ML_PREDICTIONS, GENAI_OUTPUTS
7. Build visuals (see Dashboard section below)
```

---

## 📊 Dashboard Visuals (Power BI)

| Visual | Type | Fields Used |
|--------|------|-------------|
| Earthquakes by Severity | Bar Chart | SEVERITY, TOTAL_EVENTS |
| Max Magnitude World Map | Filled Map | REGION, MAX_MAGNITUDE |
| Anomaly Detection | Donut Chart | IS_ANOMALY count |
| Severity Prediction | Pie Chart | PREDICTED_SEVERITY |
| Hourly Activity Trend | Line Chart | EVENT_HOUR, TOTAL_EVENTS |
| GenAI Summaries | Table | OUTPUT_TYPE, CONTENT |
| Top Earthquake Events | Table | PLACE, MAGNITUDE, SEVERITY |

---

## 🖼️ Sample Outputs

<img width="1315" height="753" alt="Screenshot 2026-03-01 220626" src="https://github.com/user-attachments/assets/e300cd1e-c5ef-473a-b33c-93796f6e48f0" />

<img width="1319" height="736" alt="Screenshot 2026-03-01 220638" src="https://github.com/user-attachments/assets/30e4af48-57ba-4066-a421-b326d3d972c2" />

More Coming soon :)

## 💼 Business Value

### Problem Statement
Emergency management agencies, insurance companies, research institutions, and governments need real-time awareness of seismic activity globally. Traditional approaches involve manual monitoring of reports with significant lag time between event occurrence and human awareness, limiting response effectiveness.

### Solution Value

**1. Real-Time Situational Awareness**
- Earthquake data ingested within 60 seconds of USGS publication
- Continuous streaming pipeline eliminates manual checking
- 24/7 automated monitoring with zero human intervention required

**2. Intelligent Risk Stratification**
- ML model automatically classifies earthquake severity (MICRO → CATASTROPHIC)
- Z-score anomaly detection flags statistically unusual events for immediate attention
- Removes the cognitive burden of analysts manually triaging hundreds of daily events

**3. AI-Powered Decision Support**
- GPT-4o generates professional executive summaries in seconds
- Natural language Q&A allows non-technical stakeholders to query data conversationally
- Auto-generated emergency response playbooks reduce response planning time from hours to seconds

**4. Scalable Data Architecture**
- Medallion architecture separates raw ingestion from business logic
- Delta Lake provides ACID transactions, time travel, and schema evolution
- Snowflake enables unlimited concurrent BI consumers without impacting the pipeline

**5. Quantifiable Impact**

| Metric | Before | After |
|--------|--------|-------|
| Data freshness | Hours (manual reports) | 60 seconds (automated) |
| Analysis time | 2-4 hours per event | Instant (ML + GenAI) |
| Report generation | 1-2 days | Seconds (GPT-4o) |
| Coverage | Business hours only | 24/7/365 |
| Scalability | Limited by analyst headcount | Unlimited (cloud-native) |

### Who Benefits
- **Emergency Management Agencies** — faster evacuation and response decisions
- **Insurance Companies** — instant exposure assessment for affected regions
- **Infrastructure Operators** — automated alerts for critical asset proximity to earthquakes
- **Research Institutions** — clean, structured seismic data at scale
- **News Organizations** — real-time verified earthquake intelligence

---

## 🔑 Key Technical Achievements

- **Zero-downtime ingestion** using Databricks Auto Loader with checkpointing
- **Exactly-once processing** guaranteed by Delta Lake transaction log
- **Schema evolution** handled automatically by Auto Loader cloudFiles
- **Model versioning** with MLflow experiment tracking and run IDs
- **Cross-cloud integration** — Azure (Databricks) → AWS (Snowflake) data movement
- **Production-grade error handling** with retry logic and failOnDataLoss=false

---

## 📈 Data Flow Summary

```
USGS API          → 47 earthquakes per hour (typical)
Bronze Table      → 807+ rows after several hours (raw, with duplicates)
Silver Table      → 31 unique deduplicated events
Gold Table        → 19 aggregated insight rows
ML Predictions    → 31 rows with severity + anomaly scores
GenAI Outputs     → 3 documents (summary, clusters, playbook)
Snowflake         → 3 production tables serving Power BI
```

---

## 👨‍💻 Author

Built as a portfolio project demonstrating end-to-end real-time data engineering and AI capabilities using Azure cloud services, Apache Spark, Delta Lake, and Large Language Models.

**Skills demonstrated:**
- Real-time streaming data engineering
- Medallion/Delta Lake architecture
- Machine learning with MLflow
- GenAI/LLM integration
- Cloud data warehousing (Snowflake)
- Business intelligence (Power BI)
- Azure cloud services (Databricks, ADLS Gen2, OpenAI)

---

## 📄 License

This project is for educational and portfolio purposes.
