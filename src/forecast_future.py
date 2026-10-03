import pandas as pd
import joblib


def generate_forecast(months=3):

    # ----------------------------------------
    # 1. LOAD HISTORICAL DATA
    # ----------------------------------------

    df = pd.read_csv(
        "data/processed/monthly_features.csv"
    )

    df["Order Date"] = pd.to_datetime(
        df["Order Date"]
    )

    # Sort by date
    df = df.sort_values("Order Date")

    # ----------------------------------------
    # 2. LOAD TRAINED MODEL
    # ----------------------------------------

    model = joblib.load(
        "models/sales_forecasting_model.pkl"
    )

    # ----------------------------------------
    # 3. STORE HISTORICAL SALES
    # ----------------------------------------

    sales_history = list(
        df["Sales"]
    )

    last_date = df["Order Date"].max()

    forecasts = []

    # ----------------------------------------
    # 4. GENERATE FUTURE FORECASTS
    # ----------------------------------------

    for i in range(1, months + 1):

        future_date = (
            last_date
            + pd.DateOffset(months=i)
        )

        year = future_date.year
        month = future_date.month
        quarter = future_date.quarter

        # Lag features
        lag_1 = sales_history[-1]
        lag_2 = sales_history[-2]
        lag_3 = sales_history[-3]
        lag_6 = sales_history[-6]

        # Rolling averages
        rolling_mean_3 = (
            sum(sales_history[-3:])
            / 3
        )

        rolling_mean_6 = (
            sum(sales_history[-6:])
            / 6
        )

        # ----------------------------------------
        # 5. CREATE MODEL INPUT
        # ----------------------------------------

        future_data = pd.DataFrame({
            "Year": [year],
            "Month": [month],
            "Quarter": [quarter],
            "Lag_1": [lag_1],
            "Lag_2": [lag_2],
            "Lag_3": [lag_3],
            "Lag_6": [lag_6],
            "Rolling_Mean_3": [rolling_mean_3],
            "Rolling_Mean_6": [rolling_mean_6]
        })

        # ----------------------------------------
        # 6. PREDICT
        # ----------------------------------------

        prediction = model.predict(
            future_data
        )[0]

        # Sales cannot be negative
        prediction = max(
            0,
            prediction
        )

        forecasts.append({
            "Date": future_date,
            "Forecast": prediction
        })

        # Add prediction to history
        # This allows the next month
        # to use the previous prediction
        sales_history.append(
            prediction
        )

    # ----------------------------------------
    # 7. RETURN FORECAST DATA
    # ----------------------------------------

    forecast_df = pd.DataFrame(
        forecasts
    )

    return forecast_df

