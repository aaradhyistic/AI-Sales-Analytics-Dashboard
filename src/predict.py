import pandas as pd
import joblib


# ----------------------------------------
# LOAD MODEL
# ----------------------------------------

model = joblib.load(
    "models/sales_forecasting_model.pkl"
)


FEATURES = [
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


def forecast_future(
    historical_data,
    months=3
):

    data = historical_data.copy()

    data["Order Date"] = pd.to_datetime(
        data["Order Date"]
    )

    monthly = (
        data
        .groupby(
            data["Order Date"]
            .dt.to_period("M")
        )["Sales"]
        .sum()
        .reset_index()
    )

    monthly["Order Date"] = (
        monthly["Order Date"]
        .dt.to_timestamp()
    )

    monthly = monthly.sort_values(
        "Order Date"
    )

    sales_history = (
        monthly["Sales"]
        .tolist()
    )

    last_date = monthly[
        "Order Date"
    ].max()

    predictions = []

    for i in range(months):

        future_date = (
            last_date
            + pd.DateOffset(months=i + 1)
        )

        lag_1 = sales_history[-1]

        lag_2 = sales_history[-2]

        lag_3 = sales_history[-3]

        lag_6 = sales_history[-6]

        rolling_3 = sum(
            sales_history[-3:]
        ) / 3

        rolling_6 = sum(
            sales_history[-6:]
        ) / 6

        features = pd.DataFrame([{
            "Year": future_date.year,
            "Month": future_date.month,
            "Quarter": future_date.quarter,
            "Lag_1": lag_1,
            "Lag_2": lag_2,
            "Lag_3": lag_3,
            "Lag_6": lag_6,
            "Rolling_Mean_3": rolling_3,
            "Rolling_Mean_6": rolling_6
        }])

        prediction = model.predict(
            features[FEATURES]
        )[0]

        prediction = max(
            0,
            prediction
        )

        sales_history.append(
            prediction
        )

        predictions.append({
            "Date": future_date,
            "Predicted Sales": prediction
        })

    return pd.DataFrame(
        predictions
    )