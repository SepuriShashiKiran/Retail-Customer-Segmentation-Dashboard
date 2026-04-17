import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ======================
# LOAD FILES
# ======================
kmeans = joblib.load("model/kmeans.pkl")
scaler = joblib.load("model/scaler.pkl")

df = pd.read_csv("data/customer_segments.csv")
products = pd.read_csv("data/top_products_by_segment.csv")

# ======================
# SEGMENT MAPPING
# ======================
segment_map = {
    1: "💎 High Value",
    3: "🛍️ Regular",
    2: "⚠️ At Risk",
    0: "🆕 New / Low Value"
}

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Customer Segmentation BI", layout="wide")

st.title("📊 Retail Customer Segmentation Dashboard")

# ======================
# SIDEBAR INPUT
# ======================
st.sidebar.header("Customer Input")

recency = st.sidebar.number_input("Recency (days since last purchase)", 0, 400, 30)
frequency = st.sidebar.number_input("Frequency (number of purchases)", 1, 300, 2)
monetary = st.sidebar.number_input("Monetary (total spend)", 1, 300000, 500)

# ======================
# RFM EXPLANATION
# ======================
st.markdown("""
### ℹ️ What is RFM?

- **Recency** → How recently the customer purchased  
- **Frequency** → How often they purchase  
- **Monetary** → How much they spend  

👉 These help identify customer value and behavior.
""")

# ======================
# PREDICTION FUNCTION
# ======================
def predict_customer(r, f, m):
    avg_order = m / f
    data = np.array([[r, f, m, avg_order]])
    data_log = np.log1p(data)
    data_scaled = scaler.transform(data_log)
    cluster = kmeans.predict(data_scaled)[0]
    return cluster, avg_order

cluster, avg_order = predict_customer(recency, frequency, monetary)
segment = segment_map[cluster]

# ======================
# MAIN LAYOUT
# ======================
col1, col2 = st.columns([1, 2])

# ======================
# CUSTOMER PROFILE CARD
# ======================
with col1:
    st.subheader("Customer Profile")

    st.metric("Segment", segment)
    st.metric("Avg Order Value", f"{avg_order:.2f}")

    if segment == "💎 High Value":
        st.success("High revenue, frequent and loyal customer.")
    elif segment == "🛍️ Regular":
        st.info("Consistent customer with moderate spend.")
    elif segment == "⚠️ At Risk":
        st.warning("Customer inactive — needs re-engagement.")
    else:
        st.error("Low engagement, early-stage customer.")

# ======================
# DATA FOR CHARTS
# ======================
segment_counts = df['Segment'].value_counts().reset_index()
segment_counts.columns = ['Segment','Count']

segment_revenue = df.groupby('Segment')['Monetary'].sum().reset_index()

# ======================
# PIE CHARTS
# ======================
with col2:
    st.subheader("Segment Intelligence")

    c1, c2 = st.columns(2)

    with c1:
        fig1 = px.pie(segment_counts, values='Count', names='Segment',
                      title="Customer Distribution")
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        fig2 = px.pie(segment_revenue, values='Monetary', names='Segment',
                      title="Revenue Contribution")
        st.plotly_chart(fig2, use_container_width=True)

# ======================
# SCATTER PLOT (INTERACTIVE)
# ======================
st.subheader("Customer Behavior Analysis")

fig = px.scatter(
    df,
    x="Frequency",
    y="Monetary",
    color="Segment",
    opacity=0.6,
    title="Frequency vs Monetary"
)

# highlight input customer
fig.add_scatter(
    x=[frequency],
    y=[monetary],
    mode='markers',
    marker=dict(size=12, color='black'),
    name="This Customer"
)

st.plotly_chart(fig, use_container_width=True)

# ======================
# RECOMMENDED PRODUCTS
# ======================
st.subheader("Top Products for this Segment")

top_items = products[products['Segment'] == segment].head(5)

st.dataframe(top_items[['Description','TotalPrice']])

# ======================
# BUSINESS INSIGHTS
# ======================
st.subheader("Business Insights")

if segment == "💎 High Value":
    st.write("- Offer loyalty rewards")
    st.write("- Provide premium products")
elif segment == "🛍️ Regular":
    st.write("- Upsell with bundles")
    st.write("- Encourage repeat purchases")
elif segment == "⚠️ At Risk":
    st.write("- Send discounts or reminders")
    st.write("- Re-engagement campaigns")
else:
    st.write("- Onboard with offers")
    st.write("- Encourage second purchase")

# ======================
# CORRELATION HEATMAP
# ======================
st.subheader("Feature Correlation")

corr = df[['Recency','Frequency','Monetary']].corr()

fig_corr = px.imshow(corr, text_auto=True)
st.plotly_chart(fig_corr, use_container_width=True)
