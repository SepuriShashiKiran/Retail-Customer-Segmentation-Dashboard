import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Customer Intelligence", layout="wide")

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
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Prediction", "📈 Deep Insights"])

# =========================================================
# 📊 TAB 1: OVERVIEW
# =========================================================
with tab1:

    st.subheader("📊 Business Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", total_customers)
    col2.metric("Total Revenue", f"{int(total_revenue):,}")
    col3.metric("Active Segment", segment)

    # Distribution
    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(df, names="Segment", title="Customer Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        rev_df = df.groupby('Segment')['Monetary'].sum().reset_index()
        fig = px.pie(rev_df, names="Segment", values="Monetary",
                     title="Revenue Contribution")
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 🔍 TAB 2: PREDICTION
# =========================================================
with tab2:

    st.subheader("🔍 Customer Analysis")

    # KPI CARDS
    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(f"""
    <div style="background:#1f2937;padding:20px;border-radius:10px;text-align:center">
        <h4>Segment</h4><h2>{segment}</h2>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div style="background:#1f2937;padding:20px;border-radius:10px;text-align:center">
        <h4>Customers</h4><h2>{segment_percent:.2f}%</h2>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div style="background:#1f2937;padding:20px;border-radius:10px;text-align:center">
        <h4>Revenue</h4><h2>{revenue_percent:.2f}%</h2>
    </div>
    """, unsafe_allow_html=True)

    col4.markdown(f"""
    <div style="background:#1f2937;padding:20px;border-radius:10px;text-align:center">
        <h4>Avg Spend</h4><h2>{int(segment_df['Monetary'].mean())}</h2>
    </div>
    """, unsafe_allow_html=True)

    # SCATTER
    st.subheader("📍 Your Position")

    fig = px.scatter(df, x="Frequency", y="Monetary",
                     color="Segment", opacity=0.4)

    fig.add_scatter(
        x=[frequency],
        y=[monetary],
        mode="markers+text",
        text=["YOU"],
        marker=dict(size=20, color="yellow", line=dict(width=3, color="black")),
        name="You"
    )

    st.plotly_chart(fig, use_container_width=True)

    # PERSONAL INSIGHTS
    st.subheader("🧠 Smart Insights")

    if monetary > segment_df['Monetary'].mean():
        st.success("💰 You spend MORE than your segment")
    else:
        st.warning("💰 You spend LESS than your segment")

    if frequency > segment_df['Frequency'].mean():
        st.success("🔁 Higher purchase frequency than peers")
    else:
        st.warning("🔁 Lower frequency than peers")

    if recency > segment_df['Recency'].mean():
        st.error("⚠️ Becoming inactive compared to segment")
    else:
        st.success("✅ Active customer")

# =========================================================
# 📈 TAB 3: DEEP INSIGHTS
# =========================================================
with tab3:

    st.subheader("📈 Segment Value Analysis")

    summary = df.groupby('Segment').agg({
        'CustomerID':'count',
        'Monetary':'sum'
    }).reset_index()

    summary.columns = ['Segment','Customers','Revenue']

    summary['Customer %'] = summary['Customers'] / summary['Customers'].sum()
    summary['Revenue %'] = summary['Revenue'] / summary['Revenue'].sum()

    fig = px.bar(summary, x="Segment",
                 y=["Customer %","Revenue %"],
                 barmode="group")

    st.plotly_chart(fig, use_container_width=True)

    # HEATMAP
    st.subheader("🔥 Feature Heatmap")

    heatmap_data = df.groupby('Segment')[['Recency','Frequency','Monetary']].mean()

    fig, ax = plt.subplots()
    sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # DISTRIBUTION
    st.subheader("📊 Spend Distribution")

    fig = px.histogram(segment_df, x="Monetary", nbins=50)

    fig.add_vline(
        x=monetary,
        line_width=3,
        line_dash="dash",
        line_color="red"
    )

    st.plotly_chart(fig, use_container_width=True)
