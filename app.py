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
# RFM EXPLANATION (USER FRIENDLY)
# -----------------------------
with st.expander("ℹ️ What do these inputs mean? (Simple Explanation)"):
    st.markdown("""
    **Recency (R)** → How recently a customer made a purchase  
    - Lower = very recent (good)  
    - Higher = long time ago (risk)

    **Frequency (F)** → How often the customer buys  
    - Higher = frequent buyer  
    - Lower = occasional buyer  

    **Monetary (M)** → Total money spent  
    - Higher = valuable customer  
    - Lower = low-value customer  

    👉 Together, these help businesses understand:
    - Who are their best customers  
    - Who might stop buying  
    - Who needs engagement  
    """)

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

recency = st.sidebar.slider("Recency (days)", 0, 365, 50)
frequency = st.sidebar.slider("Frequency", 1, 50, 5)
monetary = st.sidebar.slider("Monetary Spend", 0, 20000, 1000)

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
# 📊 OVERVIEW
# =========================================================
with tab1:

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", total_customers)
    col2.metric("Total Revenue", f"{int(total_revenue):,}")
    col3.metric("Your Segment", segment)

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
# 🔍 PREDICTION
# =========================================================
with tab2:

    st.subheader("📊 Segment Intelligence")

    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(f"""
    <div style="background:#4CAF50;padding:20px;border-radius:12px;text-align:center;color:white">
        <h4>Segment</h4><h2>{segment}</h2>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div style="background:#2196F3;padding:20px;border-radius:12px;text-align:center;color:white">
        <h4>% Customers</h4><h2>{segment_percent:.2f}%</h2>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div style="background:#FF9800;padding:20px;border-radius:12px;text-align:center;color:white">
        <h4>% Revenue</h4><h2>{revenue_percent:.2f}%</h2>
    </div>
    """, unsafe_allow_html=True)

    col4.markdown(f"""
    <div style="background:#9C27B0;padding:20px;border-radius:12px;text-align:center;color:white">
        <h4>Avg Spend</h4><h2>{int(segment_df['Monetary'].mean())}</h2>
    </div>
    """, unsafe_allow_html=True)

    # Scatter
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

    # Insights
    st.subheader("🧠 Smart Insights")

    if monetary > segment_df['Monetary'].mean():
        st.success("💰 You spend MORE than your segment average")
    else:
        st.warning("💰 You spend LESS than your segment average")

    if frequency > segment_df['Frequency'].mean():
        st.success("🔁 You purchase MORE frequently")
    else:
        st.warning("🔁 Your purchase frequency is lower")

    if recency > segment_df['Recency'].mean():
        st.error("⚠️ You are becoming inactive")
    else:
        st.success("✅ You are an active customer")

# =========================================================
# 📈 DEEP INSIGHTS
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

    st.subheader("🔥 Feature Heatmap")

    heatmap_data = df.groupby('Segment')[['Recency','Frequency','Monetary']].mean()

    fig, ax = plt.subplots()
    sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("📊 Spend Distribution")

    fig = px.histogram(segment_df, x="Monetary", nbins=50)

    fig.add_vline(
        x=monetary,
        line_width=3,
        line_dash="dash",
        line_color="red"
    )

    st.plotly_chart(fig, use_container_width=True)
