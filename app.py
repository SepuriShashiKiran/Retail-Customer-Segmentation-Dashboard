import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="Customer Intelligence Dashboard", layout="wide")

# ======================
# LOAD DATA
# ======================
kmeans = joblib.load("model/kmeans.pkl")
scaler = joblib.load("model/scaler.pkl")

df = pd.read_csv("data/customer_segments.csv")
products = pd.read_csv("data/top_products_by_segment.csv")

# ======================
# SEGMENT MAP + COLORS
# ======================
segment_map = {
    1: "💎 High Value",
    3: "🛍️ Regular",
    2: "⚠️ At Risk",
    0: "🆕 New / Low Value"
}

segment_colors = {
    "💎 High Value": "#2E86C1",
    "🛍️ Regular": "#28B463",
    "⚠️ At Risk": "#E67E22",
    "🆕 New / Low Value": "#7D3C98"
}

# ======================
# SIDEBAR INPUT
# ======================
st.sidebar.title("Customer Input")

recency = st.sidebar.slider("Recency (days)", 0, 400, 30)
frequency = st.sidebar.slider("Frequency", 1, 300, 2)
monetary = st.sidebar.slider("Monetary", 1, 300000, 500)

st.sidebar.markdown("""
### ℹ️ RFM Explained
- **Recency** → How recently customer purchased  
- **Frequency** → How often  
- **Monetary** → How much spent  
""")

# ======================
# PREDICTION
# ======================
def predict(r, f, m):
    avg = m / f
    data = np.array([[r, f, m, avg]])
    data_log = np.log1p(data)
    data_scaled = scaler.transform(data_log)
    cluster = kmeans.predict(data_scaled)[0]
    return cluster, avg

cluster, avg_order = predict(recency, frequency, monetary)
segment = segment_map[cluster]

# ======================
# HEADER
# ======================
st.title("📊 Customer Intelligence Dashboard")
st.markdown("Real-time segmentation with business insights")

# ======================
# KPI CALCULATIONS
# ======================
segment_counts = df['Segment'].value_counts(normalize=True) * 100
segment_revenue = df.groupby('Segment')['Monetary'].sum()
total_revenue = segment_revenue.sum()

customer_pct = segment_counts[segment]
revenue_pct = (segment_revenue[segment] / total_revenue) * 100

# ======================
# KPI CARDS
# ======================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div style="background-color:{segment_colors[segment]};padding:15px;border-radius:10px;color:white">
        <h4>Segment</h4>
        <h2>{segment}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.metric("Customer %", f"{customer_pct:.2f}%")

with col3:
    st.metric("Revenue Contribution", f"{revenue_pct:.2f}%")

# ======================
# PROFILE + INSIGHTS
# ======================
colA, colB = st.columns([1,2])

with colA:
    st.subheader("Customer Profile")
    st.metric("Recency", recency)
    st.metric("Frequency", frequency)
    st.metric("Monetary", monetary)
    st.metric("Avg Order Value", f"{avg_order:.2f}")

with colB:
    st.subheader("Behavior Insight")

    if segment == "💎 High Value":
        st.success("High-value, frequent buyer. Core revenue driver.")
    elif segment == "🛍️ Regular":
        st.info("Consistent customer with growth potential.")
    elif segment == "⚠️ At Risk":
        st.warning("Previously valuable, now inactive.")
    else:
        st.error("Low engagement, early-stage customer.")

# ======================
# PIE CHARTS
# ======================
st.subheader("Segment Overview")

c1, c2 = st.columns(2)

with c1:
    fig1 = px.pie(
        df, names='Segment',
        title="Customer Distribution",
        color='Segment',
        color_discrete_map=segment_colors
    )
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    fig2 = px.pie(
        df, values='Monetary', names='Segment',
        title="Revenue Contribution",
        color='Segment',
        color_discrete_map=segment_colors
    )
    st.plotly_chart(fig2, use_container_width=True)

# ======================
# SCATTER (KEY VISUAL)
# ======================
st.subheader("Customer Positioning")

fig = px.scatter(
    df,
    x="Frequency",
    y="Monetary",
    color="Segment",
    color_discrete_map=segment_colors,
    opacity=0.5
)

fig.add_scatter(
    x=[frequency],
    y=[monetary],
    mode='markers',
    marker=dict(size=14, color='black'),
    name="This Customer"
)

st.plotly_chart(fig, use_container_width=True)

# ======================
# PRODUCT INTELLIGENCE
# ======================
st.subheader("Likely Products This Customer May Buy")

top_items = products[products['Segment'] == segment].head(5)

fig_prod = px.bar(
    top_items,
    x='TotalPrice',
    y='Description',
    orientation='h',
    title="Top Products by Revenue",
    color_discrete_sequence=[segment_colors[segment]]
)

st.plotly_chart(fig_prod, use_container_width=True)

# ======================
# BUSINESS ACTIONS
# ======================
st.subheader("Recommended Business Actions")

if segment == "💎 High Value":
    st.write("• Loyalty programs")
    st.write("• Exclusive offers")
    st.write("• Premium bundles")

elif segment == "🛍️ Regular":
    st.write("• Upsell bundles")
    st.write("• Incentivize frequency")
    st.write("• Targeted promotions")

elif segment == "⚠️ At Risk":
    st.write("• Re-engagement emails")
    st.write("• Discounts")
    st.write("• Reminder campaigns")

else:
    st.write("• First-time offers")
    st.write("• Onboarding discounts")
    st.write("• Encourage repeat purchase")

# ======================
# CORRELATION HEATMAP
# ======================
st.subheader("Feature Relationships")

corr = df[['Recency','Frequency','Monetary']].corr()

fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='Blues')
st.plotly_chart(fig_corr, use_container_width=True)
