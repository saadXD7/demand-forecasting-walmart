import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Set up the webpage
st.set_page_config(page_title="Walmart Demand Forecasting", layout="wide")

st.title("🛒 Walmart Demand Forecasting System")
st.markdown("Built by a Data Scientist. Predicting future retail demand using Meta's Prophet AI model.")
st.markdown("---")

@st.cache_data
def load_data():
    # Load historical data
    hist = pd.read_csv('data/processed/clean_walmart_data.csv')
    hist_grouped = hist.groupby('Date')['Weekly_Sales'].sum().reset_index()
    hist_grouped['Date'] = pd.to_datetime(hist_grouped['Date'])
    
    # Load forecast data
    forecast = pd.read_csv('data/models/prophet_forecast.csv')
    forecast['ds'] = pd.to_datetime(forecast['ds'])
    
    return hist_grouped, forecast

# Fetch the data
historical, forecast = load_data()

# Build the interactive graph
st.subheader("Historical Sales vs. 12-Week AI Forecast")

fig = go.Figure()

# Add Historical Data (Blue Line)
fig.add_trace(go.Scatter(
    x=historical['Date'], 
    y=historical['Weekly_Sales'], 
    mode='lines', 
    name='Historical Sales',
    line=dict(color='#1f77b4', width=2)
))

# Add Forecast Data (Orange Dashed Line)
fig.add_trace(go.Scatter(
    x=forecast['ds'], 
    y=forecast['yhat'], 
    mode='lines', 
    name='Prophet Forecast',
    line=dict(color='#ff7f0e', width=2, dash='dot')
))

# Clean up the layout
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Total Weekly Sales ($)",
    hovermode="x unified",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

# Display the chart on the webpage
st.plotly_chart(fig, use_container_width=True)

st.success("✅ Dashboard successfully connected to the Prophet AI Model.")