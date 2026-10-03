## \AI/ML Sales Analytics Dashboard

## 

## An interactive AI/ML-powered sales analytics and decision-support dashboard built using Python, Scikit-learn, SQL, Plotly and Streamlit.

## 

## \Project Overview

## 

## Traditional sales dashboards mainly display historical data through charts and reports. This project extends traditional analytics by integrating machine learning models for sales forecasting, customer segmentation and anomaly detection.

## 

## The system processes historical sales data, performs feature engineering, trains machine learning models and presents the resulting insights through an interactive Streamlit dashboard.

## 

## \Key Features

## 

## \- Sales and profit analysis

## \- Interactive sales dashboards

## \- Time-series feature engineering

## \- Sales forecasting using Random Forest Regression

## \- Customer segmentation using RFM analysis and K-Means clustering

## \- Anomaly detection using Isolation Forest

## \- Model evaluation using MAE, RMSE, R² and MAPE

## \- Feature importance analysis

## \- Actual vs predicted sales visualization

## \- Interactive Streamlit interface

## 

## \Machine Learning Components

## 

## \1. Sales Forecasting

## 

## Random Forest Regression is used to predict future sales using temporal and historical sales features such as:

## 

## \- Month

## \- Quarter

## \- Previous month sales

## \- Lagged sales

## \- Rolling averages

## 

## \2. Customer Segmentation

## 

## RFM analysis is used to represent customer purchasing behavior using:

## 

## \- Recency

## \- Frequency

## \- Monetary value

## 

## K-Means clustering is then used to group customers with similar purchasing patterns.

## 

## \3. Anomaly Detection

## 

## Isolation Forest is used to identify transactions that are statistically unusual based on selected sales-related features.

## 

## An anomaly represents an unusual observation and does not necessarily indicate fraud.

## 

## \## Technology Stack

## 

## \- Python

## \- Pandas

## \- NumPy

## \- Scikit-learn

## \- SQLite

## \- Plotly

## \- Streamlit

## \- Joblib

## 

## \Project Structure

## 

## ├── app.py

## ├── data/

## ├── database/

## ├── models/

## ├── src/

## ├── requirements.txt

## └── README.md

