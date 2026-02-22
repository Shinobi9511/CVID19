import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

st.set_page_config(page_title="COVID-19 Analytics Dashboard", layout="wide")

# =========================
# Load Data
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv(r"data/covid19_global_statistics_2026.csv")
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df

df = load_data()

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")

model = load_model()

st.title("🌍 COVID-19 Global Analytics & Prediction Dashboard")

# ======================================================
# SIDEBAR - ADVANCED DYNAMIC FILTERS
# ======================================================
st.sidebar.header("🔎 Advanced Filters")

filtered_df = df.copy()

# -------- Country Filter --------
if "country" in df.columns:
    countries = st.sidebar.multiselect(
        "Select Country",
        options=sorted(df["country"].unique()),
        default=df["country"].unique()
    )
    filtered_df = filtered_df[filtered_df["country"].isin(countries)]

# -------- Date Filter (Auto Detect) --------
date_columns = [col for col in df.columns if "date" in col]

if date_columns:
    date_col = date_columns[0]
    filtered_df[date_col] = pd.to_datetime(filtered_df[date_col])

    start_date = st.sidebar.date_input(
        "Start Date",
        filtered_df[date_col].min()
    )
    end_date = st.sidebar.date_input(
        "End Date",
        filtered_df[date_col].max()
    )

    filtered_df = filtered_df[
        (filtered_df[date_col] >= pd.to_datetime(start_date)) &
        (filtered_df[date_col] <= pd.to_datetime(end_date))
    ]

# -------- Dynamic Numeric Filters --------
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    min_val = float(df[col].min())
    max_val = float(df[col].max())

    selected_range = st.sidebar.slider(
        f"{col.replace('_',' ').title()} Range",
        min_value=min_val,
        max_value=max_val,
        value=(min_val, max_val)
    )

    filtered_df = filtered_df[
        (filtered_df[col] >= selected_range[0]) &
        (filtered_df[col] <= selected_range[1])
    ]

st.sidebar.markdown("---")
st.sidebar.write(f"Filtered Records: {filtered_df.shape[0]}")

# ======================================================
# KPI METRICS SECTION
# ======================================================
st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)

if "total_cases" in filtered_df.columns:
    col1.metric("Total Cases", f"{int(filtered_df['total_cases'].sum()):,}")

if "total_deaths" in filtered_df.columns:
    col2.metric("Total Deaths", f"{int(filtered_df['total_deaths'].sum()):,}")

if "total_recovered" in filtered_df.columns:
    col3.metric("Total Recovered", f"{int(filtered_df['total_recovered'].sum()):,}")

# ======================================================
# VISUALIZATION SECTION
# ======================================================
st.subheader("📈 Interactive Visualizations")

if "country" in filtered_df.columns and "total_cases" in filtered_df.columns:
    top_cases = (
        filtered_df.groupby("country")["total_cases"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig_cases = px.bar(
        top_cases,
        x="country",
        y="total_cases",
        title="Top 10 Countries by Total Cases"
    )
    st.plotly_chart(fig_cases, use_container_width=True)

if "country" in filtered_df.columns and "total_deaths" in filtered_df.columns:
    top_deaths = (
        filtered_df.groupby("country")["total_deaths"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig_deaths = px.bar(
        top_deaths,
        x="country",
        y="total_deaths",
        title="Top 10 Countries by Total Deaths"
    )
    st.plotly_chart(fig_deaths, use_container_width=True)

# ======================================================
# MODEL PREDICTION SECTION
# ======================================================
st.subheader("🤖 Predict Total Deaths")

with st.expander("Enter Custom Input for Prediction"):

    input_data = {}

    for col in df.drop(columns=["total_deaths"]).columns:
        if df[col].dtype == "object":
            input_data[col] = st.selectbox(
                col.replace("_", " ").title(),
                options=df[col].unique()
            )
        else:
            input_data[col] = st.number_input(
                col.replace("_", " ").title(),
                value=float(df[col].mean())
            )

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]

        st.success(f"Predicted Total Deaths: {int(prediction):,}")

# ======================================================
# DATA VIEW SECTION
# ======================================================
st.subheader("📂 View Filtered Dataset")
st.dataframe(filtered_df)


