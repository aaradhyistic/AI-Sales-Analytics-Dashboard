# AI Sales Analytics Dashboard

An AI-powered Sales Analytics and Decision Support System built using Python, Machine Learning, SQL, and Streamlit.

---

## Project Overview

Businesses generate large amounts of sales data, but analyzing this data manually can make it difficult to identify trends, predict future sales, understand customer behavior, and detect unusual transactions.

This project uses Data Analytics and Machine Learning techniques to analyze sales data and provide useful business insights.

The system includes:

- Sales forecasting
- Customer segmentation
- Anomaly detection
- Sales performance analysis
- Interactive data visualization

The application is built using Python and Streamlit with Machine Learning models for predictive and analytical tasks.

---

## Features

- Interactive sales analytics dashboard
- Sales trend and performance analysis
- Future sales forecasting using Random Forest
- Customer segmentation using RFM Analysis and K-Means
- Anomaly detection using Isolation Forest
- Model evaluation using MAE, RMSE, R², and MAPE
- Feature importance analysis
- Interactive charts and visualizations
- Data preprocessing and feature engineering
- SQL-based data management

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SQL / SQLite
- Plotly
- Streamlit
- Joblib
- VS Code

---

## Machine Learning Models

### 1. Sales Forecasting

Random Forest Regression is used to predict sales based on historical sales patterns and engineered features.

Features may include:

- Previous sales values
- Lag features
- Rolling averages
- Month
- Quarter
- Other sales-related variables

---

### 2. Customer Segmentation

Customer segmentation is performed using:

- RFM Analysis
- K-Means Clustering

RFM stands for:

- Recency – How recently a customer purchased
- Frequency – How often a customer purchased
- Monetary – How much a customer spent

Customers are grouped into different clusters based on their purchasing behavior.

---

### 3. Anomaly Detection

Isolation Forest is used to identify unusual sales transactions or records.

The model helps identify observations that differ significantly from normal sales patterns.

An anomaly does not necessarily indicate fraud; it represents a statistically unusual observation that may require further investigation.

---

## Model Evaluation

The forecasting model is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Mean Absolute Percentage Error (MAPE)

The project also provides:

- Actual vs Predicted Sales
- Feature Importance
- Customer Cluster Analysis
- Anomaly Visualization

---

## Project Structure

```bash
AI-Sales-Analytics-Dashboard/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── sales_data.csv
│
├── models/
│   └── trained_models/
│
├── src/
│   ├── clean_data.py
│   ├── database.py
│   ├── feature_engineering.py
│   ├── train_forecasting.py
│   ├── predict.py
│   ├── anomaly_detection.py
│   └── customer_segmentation.py
│
└── screenshots/
    ├── dashboard_overview.png
    ├── sales_analysis.png
    ├── forecasting.png
    ├── customer_segmentation.png
    └── anomaly_detection.png

