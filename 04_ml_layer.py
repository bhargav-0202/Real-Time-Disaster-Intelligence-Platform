# Databricks notebook source
# Cmd 1 — Install dependencies
%pip install xgboost scikit-learn mlflow pandas numpy

# COMMAND ----------

# Cmd 2 — Config
SILVER_TABLE  = "disaster_intelligence.silver.earthquakes"
ML_TABLE      = "disaster_intelligence.gold.ml_predictions"

current_user          = spark.sql("SELECT current_user()").collect()[0][0]
MLFLOW_EXPERIMENT = "/disaster_intelligence/earthquake_severity_classifier"

print(f"   MLflow path : {MLFLOW_EXPERIMENT}")
print(f"   User        : {current_user}")

# COMMAND ----------

# Cmd 3 — Create ML schema
spark.sql("CREATE SCHEMA IF NOT EXISTS disaster_intelligence.gold")


# COMMAND ----------

# Cmd 4 — Load Silver data into Pandas for ML training
import pandas as pd
import numpy as np

silver_df = spark.sql("""
    SELECT
        event_id,
        magnitude,
        depth_km,
        latitude,
        longitude,
        significance,
        felt_reports,
        cdi,
        mmi,
        gap_deg,
        rms,
        nst,
        tsunami_flag,
        severity,
        region
    FROM disaster_intelligence.silver.earthquakes
    WHERE magnitude IS NOT NULL
      AND depth_km  IS NOT NULL
""")

pdf = silver_df.toPandas()

# Fill nulls with sensible defaults
pdf["felt_reports"] = pdf["felt_reports"].fillna(0)
pdf["cdi"]          = pdf["cdi"].fillna(0)
pdf["mmi"]          = pdf["mmi"].fillna(0)
pdf["gap_deg"]      = pdf["gap_deg"].fillna(180)
pdf["rms"]          = pdf["rms"].fillna(0)
pdf["nst"]          = pdf["nst"].fillna(0)
pdf["significance"] = pdf["significance"].fillna(0)
pdf["tsunami_flag"] = pdf["tsunami_flag"].astype(int)

print(f"Loaded {len(pdf)} rows from Silver table")
print(f"   Severity distribution:")
print(pdf["severity"].value_counts().to_string())

# COMMAND ----------

# Cmd 5 — Anomaly Detection using Z-Score
# Flag earthquakes that are statistically unusual

from scipy import stats

print("=" * 50)
print("STEP 1 — Z-Score Anomaly Detection")
print("=" * 50)

# Calculate Z-scores for key numeric features
features_for_anomaly = ["magnitude", "depth_km", "significance"]

for feat in features_for_anomaly:
    pdf[f"zscore_{feat}"] = np.abs(stats.zscore(pdf[feat].fillna(pdf[feat].mean())))

# Flag as anomaly if ANY feature Z-score > 2.5
pdf["is_anomaly"] = (
    (pdf["zscore_magnitude"]    > 2.5) |
    (pdf["zscore_depth_km"]     > 2.5) |
    (pdf["zscore_significance"] > 2.5)
).astype(int)

pdf["anomaly_reason"] = ""
pdf.loc[pdf["zscore_magnitude"]    > 2.5, "anomaly_reason"] += "unusual_magnitude "
pdf.loc[pdf["zscore_depth_km"]     > 2.5, "anomaly_reason"] += "unusual_depth "
pdf.loc[pdf["zscore_significance"] > 2.5, "anomaly_reason"] += "high_significance "
pdf["anomaly_reason"] = pdf["anomaly_reason"].str.strip()
pdf.loc[pdf["is_anomaly"] == 0, "anomaly_reason"] = "none"

anomaly_count = pdf["is_anomaly"].sum()
print(f"\n Anomaly detection complete")
print(f"   Total events    : {len(pdf)}")
print(f"   Anomalies found : {anomaly_count}")
print(f"\n   Anomalous events:")
print(pdf[pdf["is_anomaly"] == 1][
    ["event_id", "magnitude", "depth_km", "significance", "anomaly_reason"]
].to_string(index=False))

# COMMAND ----------

# Cmd 6 — Severity Classifier using XGBoost + MLflow tracking

import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore")

print("=" * 50)
print("STEP 2 — XGBoost Severity Classifier")
print("=" * 50)

# Features and target
FEATURE_COLS = [
    "magnitude", "depth_km", "latitude", "longitude",
    "significance", "felt_reports", "cdi", "mmi",
    "gap_deg", "rms", "nst", "tsunami_flag"
]

# Encode severity labels to numbers
le = LabelEncoder()
pdf["severity_encoded"] = le.fit_transform(pdf["severity"])

print(f"\nSeverity label mapping:")
for i, label in enumerate(le.classes_):
    print(f"   {i} → {label}")

X = pdf[FEATURE_COLS].values
y = pdf["severity_encoded"].values

# Split — use all data for training since dataset is small
# In production with more data, use 80/20 split
if len(pdf) >= 10:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
else:
    # Too few rows to split — train on all
    X_train, X_test, y_train, y_test = X, X, y, y
    print(" Small dataset — training and testing on same data (normal at this stage)")
    print("   As more earthquakes are collected, model will improve automatically")

print(f"\nTraining samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")

# COMMAND ----------

# Cmd 7 — Train model and log with MLflow (fixed)

import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# Set experiment — path now guaranteed to exist
mlflow.set_experiment(f"/Users/{current_user}/earthquake_severity_classifier")

with mlflow.start_run(run_name="xgboost_severity_v1") as run:

    # Train XGBoost
    model = XGBClassifier(
        n_estimators      = 100,
        max_depth         = 4,
        learning_rate     = 0.1,
        use_label_encoder = False,
        eval_metric       = "mlogloss",
        random_state      = 42,
        verbosity         = 0
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log to MLflow
    mlflow.log_param("n_estimators",  100)
    mlflow.log_param("max_depth",     4)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("features",      str(FEATURE_COLS))
    mlflow.log_param("training_rows", len(X_train))
    mlflow.log_metric("accuracy",     accuracy)

    # Log model artifact
    mlflow.sklearn.log_model(model, "xgboost_severity_model")

    RUN_ID = run.info.run_id

print(f"\n Model trained and logged to MLflow!")
print(f"   Accuracy  : {accuracy:.1%}")
print(f"   Run ID    : {RUN_ID}")
print(f"   Experiment: {MLFLOW_EXPERIMENT}")
print(f"\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    labels=list(range(len(le.classes_))),
    target_names=le.classes_,
    zero_division=0
))

print(f"\n Severity classes in your current data:")
for i, label in enumerate(le.classes_):
    count = (pdf["severity"] == label).sum()
    print(f"   {label:15} → {count} earthquakes")


# COMMAND ----------

# Cmd 8 — Feature importance
import pandas as pd

importance_df = pd.DataFrame({
    "feature"   : FEATURE_COLS,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print(" Feature Importance (what drives severity prediction):")
print(importance_df.to_string(index=False))

# COMMAND ----------

# Cmd 9 — Run predictions on ALL Silver data and save ML-enriched table

from pyspark.sql.functions import udf, col, current_timestamp, lit
from pyspark.sql.types import StringType, IntegerType, DoubleType
import mlflow.pyfunc

print("Running predictions on full Silver dataset...")

# Geting predictions for all rows
X_all    = pdf[FEATURE_COLS].values
y_all    = model.predict(X_all)
y_proba  = model.predict_proba(X_all).max(axis=1)  # confidence score

pdf["predicted_severity"]    = le.inverse_transform(y_all)
pdf["prediction_confidence"] = y_proba.round(4)
pdf["anomaly_score"]         = (
    pdf[["zscore_magnitude", "zscore_depth_km", "zscore_significance"]].max(axis=1)
).round(4)

# Converting back to Spark DataFrame
result_cols = [
    "event_id",
    "predicted_severity",
    "prediction_confidence",
    "is_anomaly",
    "anomaly_reason",
    "anomaly_score",
    "zscore_magnitude",
    "zscore_depth_km",
    "zscore_significance"
]

predictions_spark = spark.createDataFrame(pdf[result_cols])

# Join with Silver to create ML-enriched table
silver_spark = spark.table(SILVER_TABLE)

ml_enriched = (
    silver_spark
    .join(predictions_spark, on="event_id", how="left")
    .withColumn("ml_load_timestamp", current_timestamp())
    .withColumn("model_run_id",       lit(RUN_ID))
)

# Save as ML predictions table
(
    ml_enriched.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(ML_TABLE)
)

print(f" ML-enriched table saved: {ML_TABLE}")
print(f"   Total rows : {ml_enriched.count()}")

# COMMAND ----------

# Cmd 10 — Final ML validation
spark.sql(f"""
SELECT
    event_id,
    place,
    magnitude,
    severity AS actual_severity,
    predicted_severity,
    ROUND(prediction_confidence * 100, 1) AS confidence_pct,
    is_anomaly,
    anomaly_reason,
    region
FROM {ML_TABLE}
ORDER BY magnitude DESC
""").show(20, truncate=False)

# COMMAND ----------

# Cmd 11 — Anomaly summary
spark.sql(f"""
SELECT
    is_anomaly,
    anomaly_reason,
    COUNT(*)                    AS event_count,
    ROUND(AVG(magnitude), 2)    AS avg_magnitude,
    MAX(magnitude)              AS max_magnitude,
    ROUND(AVG(depth_km), 2)     AS avg_depth_km
FROM {ML_TABLE}
GROUP BY is_anomaly, anomaly_reason
ORDER BY is_anomaly DESC
""").show(truncate=False)

# COMMAND ----------

