import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("🛍️ Retail Customer Segmentation Dashboard")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/features.csv")

@st.cache_resource
def load_model():
    kmeans = joblib.load("model/kmeans.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return kmeans, scaler

df = load_data()
kmeans, scaler = load_model()

# -----------------------------
# SIDEBAR INPUT
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
# PREDICTION
# -----------------------------
if st.sidebar.button("Predict"):

    input_df = pd.DataFrame([[recency, frequency, monetary, avg_order_value]],
                            columns=['Recency','Frequency','Monetary','AvgOrderValue'])

    input_log = np.log1p(input_df)
    input_scaled = scaler.transform(input_log)

    cluster = kmeans.predict(input_scaled)[0]
    segment = segment_map[cluster]

    st.sidebar.success(f"Segment: {segment}")

    # -----------------------------
    # USER VISUALIZATION
    # -----------------------------
    st.subheader("📍 Your Position vs Customers")

    fig = px.scatter(
        df,
        x="Frequency",
        y="Monetary",
        color="Segment",
        opacity=0.5
    )

    fig.add_scatter(
        x=[frequency],
        y=[monetary],
        mode="markers",
        marker=dict(size=15, color="red"),
        name="You"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # INSIGHT
    # -----------------------------
    st.subheader("🧠 Insight")

    explanations = {
        "💎 High Value": "High spend + frequent buyer.",
        "⭐ Potential Loyal": "Can become high value.",
        "🛍️ Regular": "Moderate activity.",
        "🆕 New / Occasional": "Low engagement.",
        "⚠️ At Risk": "Inactive customer."
    }

    st.write(explanations[segment])

    # -----------------------------
    # ACTION
    # -----------------------------
    st.subheader("📈 Recommended Action")

    actions = {
        "💎 High Value": "Reward loyalty, premium offers.",
        "⭐ Potential Loyal": "Upsell and engage.",
        "🛍️ Regular": "Increase frequency.",
        "🆕 New / Occasional": "Onboard properly.",
        "⚠️ At Risk": "Re-engagement campaigns."
    }

    st.success(actions[segment])

# -----------------------------
# STATIC DASHBOARD
# -----------------------------
st.subheader("📊 Segment Distribution")

fig = px.histogram(df, x="Segment", color="Segment")
st.plotly_chart(fig, use_container_width=True)

st.subheader("📉 Recency vs Frequency")

fig = px.scatter(df, x="Recency", y="Frequency", color="Segment")
st.plotly_chart(fig, use_container_width=True)
