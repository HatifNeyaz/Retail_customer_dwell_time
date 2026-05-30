import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Retail Insights Dashboard", page_icon="🛒", layout="wide")

st.title("🛒 Retail Aisle Analytics Dashboard")
st.markdown("This dashboard reads the output of our Computer Vision tracking pipeline to provide actionable retail metrics.")

# ==========================================
# DATA LOADING
# ==========================================
# We use st.cache_data so the app doesn't reload the CSV every time you click a button
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/output/dwell_times.csv")
        return df
    except FileNotFoundError:
        return None

df = load_data()

# ==========================================
# DASHBOARD LOGIC
# ==========================================
if df is None or df.empty:
    st.warning("No tracking data found. Please run main.py first to generate the dwell_times.csv file.")
else:
    # 1. Top Level KPI Metrics
    st.subheader("Key Performance Indicators")
    col1, col2, col3 = st.columns(3)
    
    total_customers = len(df)
    avg_dwell = df["Dwell_Time_Seconds"].mean()
    max_dwell = df["Dwell_Time_Seconds"].max()
    
    with col1:
        st.metric(label="Total Engaged Customers", value=total_customers)
    with col2:
        st.metric(label="Average Dwell Time", value=f"{avg_dwell:.2f} sec")
    with col3:
        st.metric(label="Maximum Dwell Time", value=f"{max_dwell:.2f} sec")
        
    st.divider()

    # 2. Charts and Visualizations
    col_chart, col_data = st.columns([2, 1])
    
    with col_chart:
        st.subheader("Distribution of Dwell Times")
        # Create an interactive histogram using Plotly
        fig = px.histogram(
            df, 
            x="Dwell_Time_Seconds", 
            nbins=10, 
            title="How long do customers stay at the shelf?",
            labels={"Dwell_Time_Seconds": "Time Spent (Seconds)"},
            color_discrete_sequence=['#4C78A8']
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col_data:
        st.subheader("Raw Tracking Data")
        # Display the dataframe cleanly
        st.dataframe(
            df.sort_values(by="Dwell_Time_Seconds", ascending=False),
            use_container_width=True,
            hide_index=True
        )