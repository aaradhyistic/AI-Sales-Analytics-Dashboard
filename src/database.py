import pandas as pd
import sqlite3
import os

# Read cleaned data
df = pd.read_csv("data/cleaned/sales_cleaned.csv")

# Create database folder
os.makedirs("database", exist_ok=True)

# Connect to SQLite database
connection = sqlite3.connect("database/sales.db")

# Store dataframe as SQL table
df.to_sql(
    "sales",
    connection,
    if_exists="replace",
    index=False
)

print("Database created successfully.")

# Test query
query = """
SELECT COUNT(*) AS total_orders
FROM sales
"""

result = pd.read_sql_query(query, connection)

print(result)

connection.close()