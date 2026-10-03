import pandas as pd
import os


# ============================================================
# LOAD CLEANED DATA
# ============================================================

df = pd.read_csv(
    "data/cleaned/sales_cleaned.csv"
)


# ============================================================
# CONVERT DATE
# ============================================================

df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    errors="coerce"
)


# ============================================================
# CREATE MONTHLY SALES DATA
# ============================================================

monthly = (
    df.groupby(
        df["Order Date"].dt.to_period("M")
    )
    .agg({
        "Sales": "sum",
        "Profit": "sum",
        "Quantity": "sum",
        "Discount": "mean"
    })
    .reset_index()
)


monthly["Order Date"] = (
    monthly["Order Date"]
    .dt.to_timestamp()
)


monthly = monthly.sort_values(
    "Order Date"
).reset_index(drop=True)


# ============================================================
# CALENDAR FEATURES
# ============================================================

monthly["Year"] = (
    monthly["Order Date"].dt.year
)

monthly["Month"] = (
    monthly["Order Date"].dt.month
)

monthly["Quarter"] = (
    monthly["Order Date"].dt.quarter
)


# ============================================================
# LAG FEATURES
# IMPORTANT:
# These use ONLY previous months' sales.
# ============================================================

monthly["Lag_1"] = (
    monthly["Sales"].shift(1)
)

monthly["Lag_2"] = (
    monthly["Sales"].shift(2)
)

monthly["Lag_3"] = (
    monthly["Sales"].shift(3)
)

monthly["Lag_6"] = (
    monthly["Sales"].shift(6)
)


# ============================================================
# ROLLING FEATURES
# IMPORTANT:
# shift(1) ensures the current month's Sales
# is NOT included in the rolling calculation.
# ============================================================

monthly["Rolling_Mean_3"] = (
    monthly["Sales"]
    .shift(1)
    .rolling(window=3)
    .mean()
)

monthly["Rolling_Mean_6"] = (
    monthly["Sales"]
    .shift(1)
    .rolling(window=6)
    .mean()
)


# ============================================================
# REMOVE ROWS WITH MISSING VALUES
# ============================================================

monthly = monthly.dropna().reset_index(
    drop=True
)


# ============================================================
# SAVE FEATURES
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


monthly.to_csv(
    "data/processed/monthly_features.csv",
    index=False
)


print(
    "Monthly forecasting features created successfully!"
)

print(
    "Rows:",
    len(monthly)
)

print(
    "Columns:",
    list(monthly.columns)
)