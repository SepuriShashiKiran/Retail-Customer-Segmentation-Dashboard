import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("🛍️ Retail Customer Intelligence Dashboard")

# -----------------------------
# LOAD
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/features.csv")

@st.cache_resource
def load_model():
    return joblib.load("model/kmeans.pkl"), joblib.load("model/scaler.pkl")

df = load_data()
kmeans, scaler = load_model()

# -----------------------------
# INPUT
# -----------------------------
st.sidebar.header("🔍 Customer Input")

recency = st.sidebar.slider("Recency", 0, 365, 50)
frequency = st.sidebar.slider("Frequency", 1, 50, 5)
monetary = st.sidebar.slider("Monetary", 0, 20000, 1000)

avg_order_value = monetary / frequency

# -----------------------------
# SEGMENT MAP
# -----------------------------
segment_map = {
    0: "⚠️ At Risk",
    1: "🛍️ Regular",
    2: "💎 High Value",
    3: "🆕 New / Occasional",
    4: "⭐ Potential Loyal"
}

# -----------------------------
# PREDICT
# -----------------------------
input_df = pd.DataFrame([[recency, frequency, monetary, avg_order_value]],
                        columns=['Recency','Frequency','Monetary','AvgOrderValue'])

input_scaled = scaler.transform(np.log1p(input_df))
cluster = kmeans.predict(input_scaled)[0]
segment = segment_map[cluster]

st.sidebar.success(f"Segment: {segment}")

# -----------------------------
# FILTER DATA
# -----------------------------
segment_df = df[df['Segment'] == segment]

total_customers = len(df)
segment_customers = len(segment_df)

segment_percent = (segment_customers / total_customers) * 100

total_revenue = df['Monetary'].sum()
segment_revenue = segment_df['Monetary'].sum()

revenue_percent = (segment_revenue / total_revenue) * 100

# -----------------------------
# KPI CARDS
# -----------------------------
st.subheader("📊 Segment Intelligence")

col1, col2, col3 = st.columns(3)

col1.metric("Segment", segment)
col2.metric("% Customers", f"{segment_percent:.2f}%")
col3.metric("% Revenue Contribution", f"{revenue_percent:.2f}%")

# -----------------------------
# PIE CHARTS
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    fig = px.pie(
        df,
        names="Segment",
        title="Customer Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    revenue_df = df.groupby('Segment')['Monetary'].sum().reset_index()

    fig = px.pie(
        revenue_df,
        names="Segment",
        values="Monetary",
        title="Revenue Contribution by Segment"
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# SCATTER WITH USER POINT
# -----------------------------
st.subheader("📍 Your Position in Market")

fig = px.scatter(
    df,
    x="Frequency",
    y="Monetary",
    color="Segment",
    opacity=0.4
)

# CUSTOM ICON FOR USER
fig.add_scatter(
    x=[frequency],
    y=[monetary],
    mode="markers",
    marker=dict(
        size=18,
        symbol="star",
        color="gold",
        line=dict(width=2, color="black")
    ),
    name="⭐ You"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# SEGMENT COMPARISON
# -----------------------------
st.subheader("📊 You vs Segment Average")

avg_vals = segment_df[['Recency','Frequency','Monetary']].mean()

compare_df = pd.DataFrame({
    "Metric": ["Recency", "Frequency", "Monetary"],
    "You": [recency, frequency, monetary],
    "Segment Avg": [
        avg_vals['Recency'],
        avg_vals['Frequency'],
        avg_vals['Monetary']
    ]
})

fig = px.bar(compare_df, x="Metric", y=["You","Segment Avg"], barmode="group")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# INSIGHTS
# -----------------------------
st.subheader("🧠 Business Insight")

insights = {
    "💎 High Value": "You are among top customers. Strong retention strategy needed.",
    "⭐ Potential Loyal": "You are growing. Push toward loyalty.",
    "🛍️ Regular": "Stable customer. Increase engagement.",
    "🆕 New / Occasional": "Low engagement. Improve onboarding.",
    "⚠️ At Risk": "High churn risk. Immediate action needed."
}

st.info(insights[segment])

# -----------------------------
# ACTIONS
# -----------------------------
st.subheader("📈 Recommended Action")

actions = {
    "💎 High Value": "VIP rewards, premium offers",
    "⭐ Potential Loyal": "Upsell, loyalty programs",
    "🛍️ Regular": "Increase frequency via campaigns",
    "🆕 New / Occasional": "Onboarding offers",
    "⚠️ At Risk": "Discounts & reactivation campaigns"
}

st.success(actions[segment])
