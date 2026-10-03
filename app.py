import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go

from src.forecast_future import generate_forecast


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# LOAD DATABASE
# ============================================================

@st.cache_data
def load_data():

    connection = sqlite3.connect(
        "database/sales.db"
    )

    data = pd.read_sql_query(
        "SELECT * FROM sales",
        connection
    )

    connection.close()

    return data


df = load_data()


# ============================================================
# LOAD CUSTOMER SEGMENTATION
# ============================================================

try:

    customer_segments = pd.read_csv(
        "data/processed/customer_segments.csv"
    )

except FileNotFoundError:

    customer_segments = pd.DataFrame()


# ============================================================
# LOAD ANOMALY DATA
# ============================================================

try:

    anomaly_df = pd.read_csv(
        "data/processed/sales_with_anomalies.csv"
    )

except FileNotFoundError:

    anomaly_df = pd.DataFrame()


# ============================================================
# CONVERT DATE
# ============================================================

df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    errors="coerce"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "📊 Sales Analytics Dashboard"
)

st.write(
    "Analyze sales, profit, customer behavior, "
    "future sales and unusual transactions using "
    "interactive analytics and machine learning."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Dashboard Filters"
)


# ============================================================
# REGION FILTER
# ============================================================

regions = (
    df["Region"]
    .dropna()
    .unique()
)

selected_regions = st.sidebar.multiselect(
    "Region",
    regions,
    default=list(regions)
)


# ============================================================
# CATEGORY FILTER
# ============================================================

categories = (
    df["Category"]
    .dropna()
    .unique()
)

selected_categories = st.sidebar.multiselect(
    "Category",
    categories,
    default=list(categories)
)


# ============================================================
# SEGMENT FILTER
# ============================================================

segments = (
    df["Segment"]
    .dropna()
    .unique()
)

selected_segments = st.sidebar.multiselect(
    "Segment",
    segments,
    default=list(segments)
)


# ============================================================
# DATE FILTER
# ============================================================

min_date = df["Order Date"].min().date()
max_date = df["Order Date"].max().date()

date_range = st.sidebar.date_input(
    "Date Range",
    [min_date, max_date]
)


# ============================================================
# FILTER DATA
# ============================================================

filtered_df = df[
    (df["Region"].isin(selected_regions))
    & (df["Category"].isin(selected_categories))
    & (df["Segment"].isin(selected_segments))
]


if len(date_range) == 2:

    start_date = pd.to_datetime(
        date_range[0]
    )

    end_date = pd.to_datetime(
        date_range[1]
    )

    end_date = (
        end_date
        + pd.Timedelta(days=1)
        - pd.Timedelta(seconds=1)
    )

    filtered_df = filtered_df[
        (filtered_df["Order Date"] >= start_date)
        & (filtered_df["Order Date"] <= end_date)
    ]


# ============================================================
# CREATE TABS
# ============================================================

overview_tab, sales_tab, forecast_tab, customer_tab, anomaly_tab = st.tabs(
    [
        "📊 Overview",
        "📈 Sales Analysis",
        "🔮 Forecasting",
        "👥 Customers",
        "🚨 Anomalies"
    ]
)


# ============================================================
# TAB 1 - OVERVIEW
# ============================================================

with overview_tab:

    st.header(
        "📊 Business Overview"
    )

    st.write(
        "Get a quick overview of sales and "
        "business performance."
    )

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    total_sales = filtered_df["Sales"].sum()

    total_profit = filtered_df["Profit"].sum()

    total_orders = filtered_df["Order ID"].nunique()

    total_customers = filtered_df["Customer ID"].nunique()

    profit_margin = (
        (total_profit / total_sales) * 100
        if total_sales > 0
        else 0
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Total Sales",
        f"${total_sales:,.2f}"
    )

    col2.metric(
        "Total Profit",
        f"${total_profit:,.2f}"
    )

    col3.metric(
        "Total Orders",
        f"{total_orders:,}"
    )

    col4.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col5.metric(
        "Profit Margin",
        f"{profit_margin:.2f}%"
    )

    st.divider()

    # --------------------------------------------------------
    # MONTHLY SALES
    # --------------------------------------------------------

    st.subheader(
        "Monthly Sales Trend"
    )

    monthly_sales = (
        filtered_df
        .groupby("Year Month")["Sales"]
        .sum()
        .reset_index()
    )

    fig_monthly = px.line(
        monthly_sales,
        x="Year Month",
        y="Sales",
        markers=True,
        title="Monthly Sales Trend"
    )

    st.plotly_chart(
        fig_monthly,
        width="stretch"
    )

    # --------------------------------------------------------
    # MONTHLY PROFIT
    # --------------------------------------------------------

    st.subheader(
        "Monthly Profit Trend"
    )

    monthly_profit = (
        filtered_df
        .groupby("Year Month")["Profit"]
        .sum()
        .reset_index()
    )

    fig_profit_trend = px.line(
        monthly_profit,
        x="Year Month",
        y="Profit",
        markers=True,
        title="Monthly Profit Trend"
    )

    st.plotly_chart(
        fig_profit_trend,
        width="stretch"
    )


# ============================================================
# TAB 2 - SALES ANALYSIS
# ============================================================

with sales_tab:

    st.header(
        "📈 Sales Analysis"
    )

    st.write(
        "Explore sales performance across categories, "
        "regions, products and profit."
    )

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    total_sales = filtered_df["Sales"].sum()

    total_profit = filtered_df["Profit"].sum()

    total_orders = filtered_df["Order ID"].nunique()

    average_order_value = (
        total_sales / total_orders
        if total_orders > 0
        else 0
    )

    profit_margin = (
        (total_profit / total_sales) * 100
        if total_sales > 0
        else 0
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Total Sales",
        f"${total_sales:,.2f}"
    )

    col2.metric(
        "Total Profit",
        f"${total_profit:,.2f}"
    )

    col3.metric(
        "Total Orders",
        f"{total_orders:,}"
    )

    col4.metric(
        "Average Order Value",
        f"${average_order_value:,.2f}"
    )

    col5.metric(
        "Profit Margin",
        f"{profit_margin:.2f}%"
    )

    st.divider()

    # --------------------------------------------------------
    # CATEGORY AND REGION
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    category_sales = (
        filtered_df
        .groupby("Category")["Sales"]
        .sum()
        .reset_index()
        .sort_values(
            "Sales",
            ascending=False
        )
    )

    fig_category = px.bar(
        category_sales,
        x="Category",
        y="Sales",
        title="Sales by Category",
        text_auto=True
    )

    col1.plotly_chart(
        fig_category,
        width="stretch"
    )

    region_sales = (
        filtered_df
        .groupby("Region")["Sales"]
        .sum()
        .reset_index()
        .sort_values(
            "Sales",
            ascending=False
        )
    )

    fig_region = px.bar(
        region_sales,
        x="Region",
        y="Sales",
        title="Sales by Region",
        text_auto=True
    )

    col2.plotly_chart(
        fig_region,
        width="stretch"
    )

    # --------------------------------------------------------
    # PROFIT BY CATEGORY
    # --------------------------------------------------------

    profit_category = (
        filtered_df
        .groupby("Category")["Profit"]
        .sum()
        .reset_index()
        .sort_values(
            "Profit",
            ascending=False
        )
    )

    fig_profit = px.bar(
        profit_category,
        x="Category",
        y="Profit",
        title="Profit by Category",
        text_auto=True
    )

    st.plotly_chart(
        fig_profit,
        width="stretch"
    )

    # --------------------------------------------------------
    # TOP 10 PRODUCTS
    # --------------------------------------------------------

    top_products = (
        filtered_df
        .groupby("Product Name")["Sales"]
        .sum()
        .reset_index()
        .sort_values(
            "Sales",
            ascending=False
        )
        .head(10)
    )

    fig_products = px.bar(
        top_products,
        x="Sales",
        y="Product Name",
        orientation="h",
        title="Top 10 Products"
    )

    st.plotly_chart(
        fig_products,
        width="stretch"
    )

    # --------------------------------------------------------
    # SALES VS PROFIT
    # --------------------------------------------------------

    monthly_performance = (
        filtered_df
        .groupby("Year Month")[["Sales", "Profit"]]
        .sum()
        .reset_index()
    )

    fig_performance = px.line(
        monthly_performance,
        x="Year Month",
        y=["Sales", "Profit"],
        markers=True,
        title="Sales vs Profit"
    )

    st.plotly_chart(
        fig_performance,
        width="stretch"
    )

    # --------------------------------------------------------
    # SALES BY SUB-CATEGORY
    # --------------------------------------------------------

    subcategory_sales = (
        filtered_df
        .groupby("Sub-Category")["Sales"]
        .sum()
        .reset_index()
        .sort_values(
            "Sales",
            ascending=False
        )
    )

    fig_subcategory = px.bar(
        subcategory_sales,
        x="Sub-Category",
        y="Sales",
        title="Sales by Sub-Category",
        text_auto=True
    )

    st.plotly_chart(
        fig_subcategory,
        width="stretch"
    )

    # --------------------------------------------------------
    # DETAILED DATA
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Detailed Sales Data"
    )

    st.dataframe(
        filtered_df,
        width="stretch"
    )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    csv = filtered_df.to_csv(
        index=False
    )

    st.download_button(
        label="⬇️ Download Filtered Data",
        data=csv,
        file_name="filtered_sales_data.csv",
        mime="text/csv"
    )


# ============================================================
# TAB 3 - FORECASTING
# ============================================================

with forecast_tab:

    st.header(
        "🔮 Sales Forecasting"
    )

    st.write(
        "Future sales are currently predicted using "
        "the trained Random Forest model. The performance "
        "of Linear Regression, Random Forest and XGBoost "
        "is compared below."
    )

    # --------------------------------------------------------
    # FORECAST HORIZON
    # --------------------------------------------------------

    forecast_horizon = st.selectbox(
        "Forecast Horizon",
        [3, 6, 12],
        format_func=lambda x: f"{x} months"
    )

    # --------------------------------------------------------
    # FUTURE FORECAST
    # --------------------------------------------------------

    st.subheader(
        "Future Sales Forecast"
    )

    try:

        forecast_df = generate_forecast(
            forecast_horizon
        )

        display_forecast = forecast_df.copy()

        display_forecast["Date"] = (
            display_forecast["Date"]
            .dt.strftime("%B %Y")
        )

        display_forecast["Forecast"] = (
            display_forecast["Forecast"]
            .round(2)
        )

        st.dataframe(
            display_forecast,
            width="stretch"
        )

        # ----------------------------------------------------
        # FORECAST GRAPH
        # ----------------------------------------------------

        fig_forecast = go.Figure()

        fig_forecast.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Forecast"],
                mode="lines+markers",
                name="Predicted Sales"
            )
        )

        fig_forecast.update_layout(
            title="Future Sales Forecast",
            xaxis_title="Month",
            yaxis_title="Sales"
        )

        st.plotly_chart(
            fig_forecast,
            width="stretch"
        )

    except Exception as e:

        st.error(
            "Unable to generate the forecast."
        )

        st.exception(e)

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    st.divider()

    st.subheader(
        "🤖 Machine Learning Model Comparison"
    )

    st.write(
        "Three regression models are evaluated using "
        "the same time-based training and testing data."
    )

    comparison_file = (
        "models/model_comparison.csv"
    )

    try:

        comparison = pd.read_csv(
            comparison_file
        )

        # ----------------------------------------------------
        # CLEAN DATA
        # ----------------------------------------------------

        comparison.columns = (
            comparison.columns
            .str.strip()
        )

        comparison["Model"] = (
            comparison["Model"]
            .astype(str)
            .str.strip()
        )

        # Convert metrics to numeric
        metric_columns = [
            "MAE",
            "RMSE",
            "R2",
            "MAPE"
        ]

        for column in metric_columns:

            if column in comparison.columns:

                comparison[column] = pd.to_numeric(
                    comparison[column],
                    errors="coerce"
                )

        # ----------------------------------------------------
        # CHECK MODELS
        # ----------------------------------------------------

        expected_models = [
            "Linear Regression",
            "Random Forest",
            "XGBoost"
        ]

        available_models = (
            comparison["Model"]
            .tolist()
        )

        missing_models = [
            model
            for model in expected_models
            if model not in available_models
        ]

        if missing_models:

            st.warning(
                "The following models are missing "
                "from model_comparison.csv:"
            )

            st.write(
                missing_models
            )

            st.info(
                "Run python src/train_forecasting.py "
                "again to regenerate the comparison file."
            )

        # ----------------------------------------------------
        # MODEL PERFORMANCE TABLE
        # ----------------------------------------------------

        st.subheader(
            "Model Performance"
        )

        st.dataframe(
            comparison,
            width="stretch"
        )

        # ----------------------------------------------------
        # RMSE
        # ----------------------------------------------------

        st.subheader(
            "RMSE Comparison"
        )

        rmse_data = comparison.dropna(
            subset=["RMSE"]
        )

        fig_rmse = px.bar(
            rmse_data,
            x="Model",
            y="RMSE",
            title="RMSE Comparison",
            text_auto=True
        )

        st.plotly_chart(
            fig_rmse,
            width="stretch"
        )

        # ----------------------------------------------------
        # R2
        # ----------------------------------------------------

        st.subheader(
            "R² Comparison"
        )

        r2_data = comparison.dropna(
            subset=["R2"]
        )

        fig_r2 = px.bar(
            r2_data,
            x="Model",
            y="R2",
            title="R² Comparison",
            text_auto=True
        )

        st.plotly_chart(
            fig_r2,
            width="stretch"
        )

        # ----------------------------------------------------
        # MAPE
        # ----------------------------------------------------

        st.subheader(
            "MAPE Comparison"
        )

        mape_data = comparison.dropna(
            subset=["MAPE"]
        )

        fig_mape = px.bar(
            mape_data,
            x="Model",
            y="MAPE",
            title="MAPE Comparison",
            text_auto=True
        )

        st.plotly_chart(
            fig_mape,
            width="stretch"
        )

    except FileNotFoundError:

        st.warning(
            "model_comparison.csv was not found."
        )

        st.info(
            "Run python src/train_forecasting.py "
            "first."
        )

    except Exception as e:

        st.error(
            "Unable to load model comparison."
        )

        st.exception(e)

    # ========================================================
    # ACTUAL VS PREDICTED
    # ========================================================

    st.divider()

    st.subheader(
        "📊 Actual vs Predicted Sales"
    )

    prediction_file = (
        "models/actual_vs_predicted.csv"
    )

    try:

        prediction_data = pd.read_csv(
            prediction_file
        )

        prediction_data.columns = (
            prediction_data.columns
            .str.strip()
        )

        fig_prediction = go.Figure()

        # ----------------------------------------------------
        # ACTUAL
        # ----------------------------------------------------

        if "Actual" in prediction_data.columns:

            fig_prediction.add_trace(
                go.Scatter(
                    x=list(
                        range(
                            len(prediction_data)
                        )
                    ),
                    y=prediction_data["Actual"],
                    mode="lines+markers",
                    name="Actual Sales"
                )
            )

        # ----------------------------------------------------
        # LINEAR REGRESSION
        # ----------------------------------------------------

        if (
            "Linear Regression"
            in prediction_data.columns
        ):

            fig_prediction.add_trace(
                go.Scatter(
                    x=list(
                        range(
                            len(prediction_data)
                        )
                    ),
                    y=prediction_data[
                        "Linear Regression"
                    ],
                    mode="lines",
                    name="Linear Regression"
                )
            )

        # ----------------------------------------------------
        # RANDOM FOREST
        # ----------------------------------------------------

        if (
            "Random Forest"
            in prediction_data.columns
        ):

            fig_prediction.add_trace(
                go.Scatter(
                    x=list(
                        range(
                            len(prediction_data)
                        )
                    ),
                    y=prediction_data[
                        "Random Forest"
                    ],
                    mode="lines",
                    name="Random Forest"
                )
            )

        # ----------------------------------------------------
        # XGBOOST
        # ----------------------------------------------------

        if (
            "XGBoost"
            in prediction_data.columns
        ):

            fig_prediction.add_trace(
                go.Scatter(
                    x=list(
                        range(
                            len(prediction_data)
                        )
                    ),
                    y=prediction_data[
                        "XGBoost"
                    ],
                    mode="lines",
                    name="XGBoost"
                )
            )

        fig_prediction.update_layout(
            xaxis_title="Test Data Point",
            yaxis_title="Sales",
            title="Actual vs Predicted Sales",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig_prediction,
            width="stretch"
        )

        # ----------------------------------------------------
        # CHECK AVAILABLE PREDICTIONS
        # ----------------------------------------------------

        prediction_models = [
            "Linear Regression",
            "Random Forest",
            "XGBoost"
        ]

        available_predictions = [
            model
            for model in prediction_models
            if model in prediction_data.columns
        ]

        st.caption(
            "Models displayed: "
            + ", ".join(available_predictions)
        )

    except FileNotFoundError:

        st.warning(
            "actual_vs_predicted.csv was not found."
        )

        st.info(
            "Run python src/train_forecasting.py "
            "first."
        )

    except Exception as e:

        st.error(
            "Unable to load actual vs predicted data."
        )

        st.exception(e)

    # ========================================================
    # RANDOM FOREST FEATURE IMPORTANCE
    # ========================================================

    st.divider()

    st.subheader(
        "🔍 Random Forest Feature Importance"
    )

    importance_file = (
        "models/feature_importance.csv"
    )

    try:

        importance_data = pd.read_csv(
            importance_file
        )

        importance_data.columns = (
            importance_data.columns
            .str.strip()
        )

        fig_importance = px.bar(
            importance_data,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Random Forest Feature Importance"
        )

        st.plotly_chart(
            fig_importance,
            width="stretch"
        )

    except FileNotFoundError:

        st.warning(
            "Random Forest feature importance "
            "file was not found."
        )

    # ========================================================
    # XGBOOST FEATURE IMPORTANCE
    # ========================================================

    st.subheader(
        "🔍 XGBoost Feature Importance"
    )

    xgb_importance_file = (
        "models/xgboost_feature_importance.csv"
    )

    try:

        xgb_importance_data = pd.read_csv(
            xgb_importance_file
        )

        xgb_importance_data.columns = (
            xgb_importance_data.columns
            .str.strip()
        )

        fig_xgb_importance = px.bar(
            xgb_importance_data,
            x="Importance",
            y="Feature",
            orientation="h",
            title="XGBoost Feature Importance"
        )

        st.plotly_chart(
            fig_xgb_importance,
            width="stretch"
        )

    except FileNotFoundError:

        st.warning(
            "XGBoost feature importance "
            "file was not found."
        )


# ============================================================
# TAB 4 - CUSTOMER SEGMENTATION
# ============================================================

with customer_tab:

    st.header(
        "👥 Customer Segmentation"
    )

    st.write(
        "Customers are grouped using RFM analysis "
        "and K-Means clustering based on Recency, "
        "Frequency and Monetary value."
    )

    # --------------------------------------------------------
    # CHECK DATA
    # --------------------------------------------------------

    if customer_segments.empty:

        st.warning(
            "Run customer_segmentation.py first "
            "to generate customer segments."
        )

    else:

        required_columns = [
            "Customer ID",
            "Recency",
            "Frequency",
            "Monetary",
            "Cluster",
            "Cluster Name"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in customer_segments.columns
        ]

        if missing_columns:

            st.error(
                "Customer segmentation file is missing "
                "the following columns:"
            )

            st.write(
                missing_columns
            )

        else:

            # ------------------------------------------------
            # CUSTOMER KPIs
            # ------------------------------------------------

            total_customers = (
                customer_segments["Customer ID"]
                .nunique()
            )

            cluster_counts = (
                customer_segments["Cluster"]
                .value_counts()
                .sort_index()
            )

            st.subheader(
                "Customer Segmentation Overview"
            )

            col1, col2, col3, col4, col5 = st.columns(5)

            col1.metric(
                "Total Customers",
                f"{total_customers:,}"
            )

            col2.metric(
                "Cluster 0",
                f"{cluster_counts.get(0, 0):,}"
            )

            col3.metric(
                "Cluster 1",
                f"{cluster_counts.get(1, 0):,}"
            )

            col4.metric(
                "Cluster 2",
                f"{cluster_counts.get(2, 0):,}"
            )

            col5.metric(
                "Cluster 3",
                f"{cluster_counts.get(3, 0):,}"
            )

            st.divider()

            # ------------------------------------------------
            # SEGMENT SUMMARY
            # ------------------------------------------------

            segment_summary = (
                customer_segments
                .groupby("Cluster Name")
                .agg({
                    "Customer ID": "count",
                    "Recency": "mean",
                    "Frequency": "mean",
                    "Monetary": "mean"
                })
                .reset_index()
            )

            segment_summary.columns = [
                "Customer Segment",
                "Number of Customers",
                "Average Recency",
                "Average Frequency",
                "Average Monetary Value"
            ]

            segment_summary[
                [
                    "Average Recency",
                    "Average Frequency",
                    "Average Monetary Value"
                ]
            ] = (
                segment_summary[
                    [
                        "Average Recency",
                        "Average Frequency",
                        "Average Monetary Value"
                    ]
                ].round(2)
            )

            st.subheader(
                "Segment Summary"
            )

            st.dataframe(
                segment_summary,
                width="stretch"
            )

            # ------------------------------------------------
            # CUSTOMERS BY SEGMENT
            # ------------------------------------------------

            fig_segments = px.bar(
                segment_summary,
                x="Customer Segment",
                y="Number of Customers",
                title="Customers by Segment",
                text_auto=True
            )

            st.plotly_chart(
                fig_segments,
                width="stretch"
            )

            # ------------------------------------------------
            # FREQUENCY VS MONETARY
            # ------------------------------------------------

            st.subheader(
                "RFM Customer Segmentation"
            )

            fig_clusters = px.scatter(
                customer_segments,
                x="Frequency",
                y="Monetary",
                color="Cluster Name",
                size="Monetary",
                hover_data=[
                    "Customer ID",
                    "Recency"
                ],
                title=(
                    "Customer Segments: "
                    "Frequency vs Monetary"
                )
            )

            fig_clusters.update_layout(
                xaxis_title="Purchase Frequency",
                yaxis_title="Monetary Value"
            )

            st.plotly_chart(
                fig_clusters,
                width="stretch"
            )

            # ------------------------------------------------
            # RECENCY VS FREQUENCY
            # ------------------------------------------------

            st.subheader(
                "Recency vs Frequency"
            )

            fig_recency = px.scatter(
                customer_segments,
                x="Recency",
                y="Frequency",
                color="Cluster Name",
                size="Monetary",
                hover_data=[
                    "Customer ID",
                    "Monetary"
                ],
                title="Recency vs Frequency by Segment"
            )

            st.plotly_chart(
                fig_recency,
                width="stretch"
            )

            # ------------------------------------------------
            # CUSTOMER DETAILS
            # ------------------------------------------------

            st.subheader(
                "Customer Segment Details"
            )

            st.dataframe(
                customer_segments,
                width="stretch"
            )


# ============================================================
# TAB 5 - ANOMALY DETECTION
# ============================================================

with anomaly_tab:

    st.header(
        "🚨 Anomaly Detection"
    )

    st.write(
        "Transactions identified as statistically "
        "unusual using Isolation Forest."
    )

    # --------------------------------------------------------
    # CHECK DATA
    # --------------------------------------------------------

    if anomaly_df.empty:

        st.warning(
            "Run anomaly_detection.py first "
            "to generate anomaly results."
        )

    elif "Anomaly Status" not in anomaly_df.columns:

        st.warning(
            "The anomaly file does not contain "
            "'Anomaly Status'. Run the updated "
            "anomaly_detection.py."
        )

    else:

        # ----------------------------------------------------
        # KPI CALCULATIONS
        # ----------------------------------------------------

        total_anomalies = (
            anomaly_df["Anomaly Status"]
            .eq("Anomaly")
            .sum()
        )

        total_records = len(
            anomaly_df
        )

        normal_records = (
            anomaly_df["Anomaly Status"]
            .eq("Normal")
            .sum()
        )

        anomaly_percentage = (
            total_anomalies
            / total_records
            * 100
            if total_records > 0
            else 0
        )

        # ----------------------------------------------------
        # KPIs
        # ----------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Transactions",
            f"{total_records:,}"
        )

        col2.metric(
            "Normal Transactions",
            f"{normal_records:,}"
        )

        col3.metric(
            "Potential Anomalies",
            f"{total_anomalies:,}"
        )

        col4.metric(
            "Anomaly Percentage",
            f"{anomaly_percentage:.2f}%"
        )

        st.divider()

        # ----------------------------------------------------
        # ANOMALY DISTRIBUTION
        # ----------------------------------------------------

        st.subheader(
            "Normal vs Anomalous Transactions"
        )

        anomaly_counts = (
            anomaly_df["Anomaly Status"]
            .value_counts()
            .reset_index()
        )

        anomaly_counts.columns = [
            "Status",
            "Count"
        ]

        fig_anomaly_count = px.bar(
            anomaly_counts,
            x="Status",
            y="Count",
            title="Transaction Anomaly Distribution",
            text_auto=True
        )

        st.plotly_chart(
            fig_anomaly_count,
            width="stretch"
        )

        # ----------------------------------------------------
        # ANOMALY TABLE
        # ----------------------------------------------------

        st.subheader(
            "Potentially Unusual Transactions"
        )

        anomalies = anomaly_df[
            anomaly_df["Anomaly Status"] == "Anomaly"
        ]

        st.dataframe(
            anomalies,
            width="stretch"
        )

        # ----------------------------------------------------
        # SALES VS PROFIT
        # ----------------------------------------------------

        if (
            "Sales" in anomaly_df.columns
            and "Profit" in anomaly_df.columns
        ):

            st.subheader(
                "Sales vs Profit - Anomaly Analysis"
            )

            hover_columns = []

            if "Order ID" in anomaly_df.columns:

                hover_columns.append(
                    "Order ID"
                )

            if "Customer ID" in anomaly_df.columns:

                hover_columns.append(
                    "Customer ID"
                )

            fig_anomaly = px.scatter(
                anomaly_df,
                x="Sales",
                y="Profit",
                color="Anomaly Status",
                hover_data=hover_columns,
                title="Normal vs Unusual Transactions"
            )

            st.plotly_chart(
                fig_anomaly,
                width="stretch"
            )

        # ----------------------------------------------------
        # SALES DISTRIBUTION
        # ----------------------------------------------------

        if "Sales" in anomaly_df.columns:

            st.subheader(
                "Sales Distribution of Transactions"
            )

            fig_sales_distribution = px.box(
                anomaly_df,
                x="Anomaly Status",
                y="Sales",
                color="Anomaly Status",
                title="Sales Distribution: Normal vs Anomaly"
            )

            st.plotly_chart(
                fig_sales_distribution,
                width="stretch"
            )
