import pandas as pd
import os

# Read raw Excel data
file_path = "data/raw/sales.xls"

df = pd.read_excel(file_path)

print("Original data:")
print(df.head())

print("\nColumns:")
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows where important values are missing
df = df.dropna(subset=["Order Date", "Sales"])

# Convert Order Date to datetime
df["Order Date"] = pd.to_datetime(df["Order Date"])

# Convert numeric columns
df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")

if "Profit" in df.columns:
    df["Profit"] = pd.to_numeric(df["Profit"], errors="coerce")

if "Quantity" in df.columns:
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

if "Discount" in df.columns:
    df["Discount"] = pd.to_numeric(df["Discount"], errors="coerce")

# Remove invalid sales values
df = df[df["Sales"] >= 0]

# Missing Values
df.isnull().sum()

# Duplicate Values 
df.isnull().sum()

# Incorrect Data types
df["Order Date"] = pd.to_datetime(df["Order Date"])

# Numerical conversion
df["Sales"] = pd.to_numeric(
    df["Sales"],
    errors="coerce"
)

# Create Year and Month columns
df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month

# Create Month Name
df["Month Name"] = df["Order Date"].dt.strftime("%B")

# Create Year-Month
df["Year Month"] = df["Order Date"].dt.to_period("M").astype(str)

# Create cleaned folder
os.makedirs("data/cleaned", exist_ok=True)

# Save cleaned data
df.to_csv(
    "data/cleaned/sales_cleaned.csv",
    index=False
)

print("\nCleaning completed.")
print("Rows after cleaning:", len(df))
print("Cleaned file saved.")