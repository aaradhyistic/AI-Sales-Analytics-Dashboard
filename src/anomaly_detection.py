import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ----------------------------------------
# 1. LOAD CLEANED DATA
# ----------------------------------------

df = pd.read_csv(
    "data/cleaned/sales_cleaned.csv"
)

# ----------------------------------------
# 2. SELECT FEATURES
# ----------------------------------------

features = [
    "Sales",
    "Quantity",
    "Discount",
    "Profit"
]

# Keep only columns that actually exist
features = [
    column for column in features
    if column in df.columns
]

print("Features used:")
print(features)


# ----------------------------------------
# 3. REMOVE MISSING VALUES
# ----------------------------------------

model_data = df[features].copy()

model_data = model_data.dropna()


# ----------------------------------------
# 4. SCALE THE FEATURES
# ----------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    model_data
)


# ----------------------------------------
# 5. CREATE ISOLATION FOREST
# ----------------------------------------

model = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42
)


# ----------------------------------------
# 6. TRAIN MODEL
# ----------------------------------------

model.fit(X_scaled)


# ----------------------------------------
# 7. PREDICT ANOMALIES
# ----------------------------------------

predictions = model.predict(X_scaled)


# ----------------------------------------
# 8. ADD RESULTS
# ----------------------------------------

model_data["Anomaly"] = predictions


# -1 = anomaly
#  1 = normal

model_data["Anomaly Status"] = np.where(
    model_data["Anomaly"] == -1,
    "Anomaly",
    "Normal"
)


# ----------------------------------------
# 9. ADD RESULTS BACK TO ORIGINAL DATA
# ----------------------------------------

df.loc[
    model_data.index,
    "Anomaly"
] = model_data["Anomaly"]

df.loc[
    model_data.index,
    "Anomaly Status"
] = model_data["Anomaly Status"]


# ----------------------------------------
# 10. SAVE RESULTS
# ----------------------------------------

df.to_csv(
    "data/processed/sales_with_anomalies.csv",
    index=False
)


# ----------------------------------------
# 11. DISPLAY RESULTS
# ----------------------------------------

total_anomalies = (
    df["Anomaly Status"] == "Anomaly"
).sum()

print("\nAnomaly Detection Completed")

print(
    "Total records:",
    len(df)
)

print(
    "Total anomalies:",
    total_anomalies
)

print(
    "Anomaly percentage:",
    round(
        (total_anomalies / len(df)) * 100,
        2
    ),
    "%"
)

print("\nSample anomalies:")

print(
    df[
        df["Anomaly Status"] == "Anomaly"
    ][features + ["Anomaly Status"]]
    .head(10)
)