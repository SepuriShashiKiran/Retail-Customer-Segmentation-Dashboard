import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="Customer Intelligence Dashboard", layout="wide")
st.markdown("""
<style>

/* -------- KPI CARDS -------- */
.kpi-card {
    padding: 18px;
    border-radius: 12px;
    color: white;
    text-align: center;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
    margin-bottom: 12px;  /* mobile spacing */
}

/* -------- PROFILE BOX -------- */
.profile-box {
    border-radius: 12px;
    padding: 20px;
    border: 1px solid rgba(128,128,128,0.2);
    background-color: rgba(0,0,0,0.03); /* works in both themes */
}

/* DARK MODE FIX */
@media (prefers-color-scheme: dark) {
    .profile-box {
        background-color: rgba(255,255,255,0.05);
    }
}

</style>
""", unsafe_allow_html=True)

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
    "💎 High Value": "#1F77B4",
    "🛍️ Regular": "#2CA02C",
    "⚠️ At Risk": "#FF7F0E",
    "🆕 New / Low Value": "#9467BD"
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
- **Monetary** → Total spend  
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
# KPI CALCULATIONS
# ======================
segment_counts = df['Segment'].value_counts(normalize=True) * 100
segment_revenue = df.groupby('Segment')['Monetary'].sum()
total_revenue = segment_revenue.sum()

customer_pct = segment_counts[segment]
revenue_pct = (segment_revenue[segment] / total_revenue) * 100

# ======================
# TITLE
# ======================
st.title("📊 Customer Intelligence Dashboard")

# ======================
# KPI CARDS (DESIGNED)
# ======================
col1, col2, col3 = st.columns(3)

def card(title, value, color):
    return f"""
    <div style="
        background-color:{color};
        padding:18px;
        border-radius:12px;
        color:white;
        text-align:center;
        box-shadow:0px 4px 12px rgba(0,0,0,0.15)">
        <h5>{title}</h5>
        <h2>{value}</h2>
    </div>
    """

col1.markdown(card("Segment", segment, segment_colors[segment]), unsafe_allow_html=True)
col2.markdown(card("Customer %", f"{customer_pct:.2f}%", "#34495E"), unsafe_allow_html=True)
col3.markdown(card("Revenue %", f"{revenue_pct:.2f}%", "#2C3E50"), unsafe_allow_html=True)

# ======================
# PROFILE BOX (CLEAN)
# ======================
st.markdown("### Customer Profile")

st.markdown(f"""
<div style="
    border:1px solid #ddd;
    border-radius:12px;
    padding:20px;
    background-color:#fafafa;
">
    <b>Recency:</b> {recency} days<br>
    <b>Frequency:</b> {frequency}<br>
    <b>Monetary:</b> {monetary}<br>
    <b>Avg Order Value:</b> {avg_order:.2f}
</div>
""", unsafe_allow_html=True)

# ======================
# INSIGHT
# ======================
st.markdown("### Behavior Insight")

if segment == "💎 High Value":
    st.success("This customer is a high-value, frequent buyer contributing significantly to revenue.")
elif segment == "🛍️ Regular":
    st.info("This customer is consistent and has strong potential to become high value.")
elif segment == "⚠️ At Risk":
    st.warning("This customer was previously active but is now at risk of churn.")
else:
    st.error("This customer has low engagement and needs nurturing.")

# ======================
# PIE CHARTS
# ======================
st.markdown("### Segment Overview")

c1, c2 = st.columns(2)

with c1:
    fig1 = px.pie(df, names='Segment', color='Segment',
                  color_discrete_map=segment_colors,
                  title="Customer Distribution")
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    fig2 = px.pie(df, values='Monetary', names='Segment',
                  color='Segment',
                  color_discrete_map=segment_colors,
                  title="Revenue Contribution")
    st.plotly_chart(fig2, use_container_width=True)

# ======================
# SCATTER (IMPROVED MARKER)
# ======================
st.markdown("### Customer Positioning")

fig = px.scatter(
    df,
    x="Frequency",
    y="Monetary",
    color="Segment",
    color_discrete_map=segment_colors,
    opacity=0.4
)

# ADD CUSTOMER ON TOP (IMPORTANT)
fig.add_trace(go.Scatter(
    x=[frequency],
    y=[monetary],
    mode='markers+text',
    marker=dict(
        size=16,
        color='black',
        symbol='diamond',
        line=dict(width=2, color='white')
    ),
    text=["This Customer"],
    textposition="top center",
    name="This Customer"
))

st.plotly_chart(fig, use_container_width=True)

# ======================
# PRODUCT INSIGHTS
# ======================
st.markdown("### Likely Products This Customer May Buy")

top_items = products[products['Segment'] == segment].head(5)

fig_prod = px.bar(
    top_items,
    x='TotalPrice',
    y='Description',
    orientation='h',
    color_discrete_sequence=[segment_colors[segment]]
)

st.plotly_chart(fig_prod, use_container_width=True)

# ======================
# BUSINESS ACTIONS
# ======================
st.markdown("### Recommended Actions")

if segment == "💎 High Value":
    st.write("• Loyalty programs • Premium offers • Exclusive deals")
elif segment == "🛍️ Regular":
    st.write("• Upsell bundles • Encourage repeat purchases")
elif segment == "⚠️ At Risk":
    st.write("• Discounts • Re-engagement campaigns")
else:
    st.write("• Onboarding offers • First-time incentives")

# ======================
# HEATMAP
# ======================
st.markdown("### Feature Relationships")

corr = df[['Recency','Frequency','Monetary']].corr()

fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='Blues')
st.plotly_chart(fig_corr, use_container_width=True)
