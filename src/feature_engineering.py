import pandas as pd

df = pd.read_csv("data/processed/sales_cleaned.csv")

df["Order Date"] = pd.to_datetime(df["Order Date"])

monthly = (
    df.groupby(
        df["Order Date"].dt.to_period("M")
    )["Sales"]
    .sum()
    .reset_index()
)

monthly["Order Date"] = monthly["Order Date"].dt.to_timestamp()

monthly["Year"] = monthly["Order Date"].dt.year

monthly["Month"] = monthly["Order Date"].dt.month

monthly["Quarter"] = monthly["Order Date"].dt.quarter

monthly["Lag_1"] = monthly["Sales"].shift(1)

monthly["Lag_2"] = monthly["Sales"].shift(2)

monthly["Lag_3"] = monthly["Sales"].shift(3)

monthly["Lag_6"] = monthly["Sales"].shift(6)

monthly["Rolling_Mean_3"] = (
    monthly["Sales"]
    .rolling(3)
    .mean()
)

monthly["Rolling_Mean_6"] = (
    monthly["Sales"]
    .rolling(6)
    .mean()
)

monthly = monthly.dropna()

monthly.to_csv(
    "data/processed/monthly_features.csv",
    index=False
)

print(monthly.head())