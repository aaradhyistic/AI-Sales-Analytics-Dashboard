import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import os


# ============================================================
# 1. LOAD CLEANED DATA
# ============================================================

df = pd.read_csv(
    "data/cleaned/sales_cleaned.csv"
)

print("Dataset loaded successfully!")
print("Rows:", len(df))


# ============================================================
# 2. CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()

print("\nAvailable columns:")
print(df.columns.tolist())


# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Customer ID",
    "Order Date",
    "Order ID",
    "Sales"
]

for column in required_columns:

    if column not in df.columns:
        raise ValueError(
            "Required column is missing: "
            + column
        )


# ============================================================
# 4. CONVERT ORDER DATE
# ============================================================

df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    errors="coerce"
)


# Remove rows where date could not be converted

df = df.dropna(
    subset=["Order Date"]
)


# ============================================================
# 5. CREATE REFERENCE DATE
# ============================================================

reference_date = (
    df["Order Date"].max()
    + pd.Timedelta(days=1)
)


# ============================================================
# 6. CREATE RFM DATA
# ============================================================

customer_data = (
    df.groupby("Customer ID")
    .agg({
        "Order Date": lambda x:
            (reference_date - x.max()).days,

        "Order ID": "nunique",

        "Sales": "sum"
    })
    .reset_index()
)


# ============================================================
# 7. RENAME RFM COLUMNS
# ============================================================

customer_data.columns = [
    "Customer ID",
    "Recency",
    "Frequency",
    "Monetary"
]


print("\nRFM data created successfully!")

print(
    customer_data.head()
)


# ============================================================
# 8. SELECT RFM FEATURES
# ============================================================

features = [
    "Recency",
    "Frequency",
    "Monetary"
]


X = customer_data[features]


# ============================================================
# 9. SCALE RFM FEATURES
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)


# ============================================================
# 10. K-MEANS CLUSTERING
# ============================================================

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

customer_data["Cluster"] = (
    kmeans.fit_predict(X_scaled)
)


# ============================================================
# 11. CLUSTER SUMMARY
# ============================================================

cluster_summary = (
    customer_data
    .groupby("Cluster")
    [["Recency", "Frequency", "Monetary"]]
    .mean()
    .round(2)
)


print("\nCluster Summary")
print("----------------------------")

print(
    cluster_summary
)


# ============================================================
# 12. CREATE CLUSTER NAMES
# ============================================================

# IMPORTANT:
# Cluster numbers are not automatically meaningful.
# We first inspect the averages above.
#
# These are initial names and can be changed
# according to your actual cluster_summary.

cluster_names = {
    0: "Segment 1",
    1: "Segment 2",
    2: "Segment 3",
    3: "Segment 4"
}


customer_data["Cluster Name"] = (
    customer_data["Cluster"]
    .map(cluster_names)
)


# ============================================================
# 13. CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


# ============================================================
# 14. SAVE CUSTOMER SEGMENTS
# ============================================================

output_file = (
    "data/processed/customer_segments.csv"
)


customer_data.to_csv(
    output_file,
    index=False
)


# ============================================================
# 15. VERIFY OUTPUT
# ============================================================

print("\nCustomer segmentation completed!")

print(
    "\nOutput columns:"
)

print(
    customer_data.columns.tolist()
)


print(
    "\nSaved successfully to:"
)

print(
    output_file
)