import pandas as pd
import numpy as np
import os
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    "data/processed/monthly_features.csv"
)


# ============================================================
# DEFINE FEATURES
# ============================================================

features = [
    "Year",
    "Month",
    "Quarter",
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Lag_6",
    "Rolling_Mean_3",
    "Rolling_Mean_6"
]

X = df[features]

y = df["Sales"]


# ============================================================
# TIME-BASED TRAIN TEST SPLIT
# ============================================================

split = int(len(df) * 0.80)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]


print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# ============================================================
# CREATE MODELS
# ============================================================

linear_model = LinearRegression()


random_forest_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)


xgboost_model = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)


# ============================================================
# TRAIN LINEAR REGRESSION
# ============================================================

print("\nTraining Linear Regression...")

linear_model.fit(
    X_train,
    y_train
)

linear_predictions = (
    linear_model.predict(X_test)
)


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

random_forest_model.fit(
    X_train,
    y_train
)

random_forest_predictions = (
    random_forest_model.predict(X_test)
)


# ============================================================
# TRAIN XGBOOST
# ============================================================

print("\nTraining XGBoost...")

xgboost_model.fit(
    X_train,
    y_train
)

xgboost_predictions = (
    xgboost_model.predict(X_test)
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    actual,
    predicted
):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = mean_squared_error(
        actual,
        predicted
    ) ** 0.5

    r2 = r2_score(
        actual,
        predicted
    )

    non_zero_actual = actual != 0

    mape = np.mean(
        np.abs(
            (
                actual[non_zero_actual]
                - predicted[non_zero_actual]
            )
            / actual[non_zero_actual]
        )
    ) * 100

    return mae, rmse, r2, mape


# ============================================================
# CALCULATE METRICS
# ============================================================

linear_mae, linear_rmse, linear_r2, linear_mape = (
    calculate_metrics(
        y_test,
        linear_predictions
    )
)


rf_mae, rf_rmse, rf_r2, rf_mape = (
    calculate_metrics(
        y_test,
        random_forest_predictions
    )
)


xgb_mae, xgb_rmse, xgb_r2, xgb_mape = (
    calculate_metrics(
        y_test,
        xgboost_predictions
    )
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n========================================")
print("MODEL COMPARISON")
print("========================================")

print("\nLinear Regression")
print("MAE:", linear_mae)
print("RMSE:", linear_rmse)
print("R2:", linear_r2)
print("MAPE:", linear_mape)

print("\nRandom Forest")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R2:", rf_r2)
print("MAPE:", rf_mape)

print("\nXGBoost")
print("MAE:", xgb_mae)
print("RMSE:", xgb_rmse)
print("R2:", xgb_r2)
print("MAPE:", xgb_mape)


# ============================================================
# CREATE MODELS FOLDER
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)


# ============================================================
# SAVE MODELS
# ============================================================

joblib.dump(
    linear_model,
    "models/linear_regression_model.pkl"
)


joblib.dump(
    random_forest_model,
    "models/random_forest_model.pkl"
)


joblib.dump(
    xgboost_model,
    "models/xgboost_model.pkl"
)


# Keep Random Forest as the existing
# default forecasting model for now.

joblib.dump(
    random_forest_model,
    "models/sales_forecasting_model.pkl"
)


# ============================================================
# MODEL COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Random Forest",
        "XGBoost"
    ],

    "MAE": [
        linear_mae,
        rf_mae,
        xgb_mae
    ],

    "RMSE": [
        linear_rmse,
        rf_rmse,
        xgb_rmse
    ],

    "R2": [
        linear_r2,
        rf_r2,
        xgb_r2
    ],

    "MAPE": [
        linear_mape,
        rf_mape,
        xgb_mape
    ]

})


comparison = comparison.round(4)


comparison.to_csv(
    "models/model_comparison.csv",
    index=False
)


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

actual_vs_predicted = pd.DataFrame({

    "Actual": y_test.values,

    "Linear Regression": (
        linear_predictions
    ),

    "Random Forest": (
        random_forest_predictions
    ),

    "XGBoost": (
        xgboost_predictions
    )

})


actual_vs_predicted.to_csv(
    "models/actual_vs_predicted.csv",
    index=False
)


# ============================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({

    "Feature": features,

    "Importance": (
        random_forest_model.feature_importances_
    )

})


feature_importance = (
    feature_importance
    .sort_values(
        "Importance",
        ascending=False
    )
)


feature_importance.to_csv(
    "models/feature_importance.csv",
    index=False
)


# ============================================================
# XGBOOST FEATURE IMPORTANCE
# ============================================================

xgb_feature_importance = pd.DataFrame({

    "Feature": features,

    "Importance": (
        xgboost_model.feature_importances_
    )

})


xgb_feature_importance = (
    xgb_feature_importance
    .sort_values(
        "Importance",
        ascending=False
    )
)


xgb_feature_importance.to_csv(
    "models/xgboost_feature_importance.csv",
    index=False
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")

print(
    "\nSaved files:"
)

print(
    "models/linear_regression_model.pkl"
)

print(
    "models/random_forest_model.pkl"
)

print(
    "models/xgboost_model.pkl"
)

print(
    "models/model_comparison.csv"
)

print(
    "models/actual_vs_predicted.csv"
)

print(
    "models/feature_importance.csv"
)

print(
    "models/xgboost_feature_importance.csv"
)