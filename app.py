import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Walmart Demand Forecasting", layout="wide")

st.title("🛒 Walmart Demand Forecasting System")
st.markdown("Built by a Data Scientist. Predicting future retail demand using Meta's Prophet AI model.")

# --- NEW FEATURE: THE EXECUTIVE CONTROL PANEL ---
st.sidebar.header("⚙️ Executive Controls")
st.sidebar.markdown("Use the slider to adjust the forecasting horizon.")
# This slider lets the user pick any number between 4 and 52
forecast_weeks = st.sidebar.slider("Weeks to Forecast", min_value=4, max_value=52, value=12, step=1)
st.markdown("---")

@st.cache_data
def load_data():
    hist = pd.read_csv('data/processed/clean_walmart_data.csv')
    hist_grouped = hist.groupby('Date')['Weekly_Sales'].sum().reset_index()
    hist_grouped['Date'] = pd.to_datetime(hist_grouped['Date'])
    
    forecast = pd.read_csv('data/models/prophet_forecast.csv')
    forecast['ds'] = pd.to_datetime(forecast['ds'])
    return hist_grouped, forecast

historical, full_forecast = load_data()

# --- NEW FEATURE: DYNAMIC FILTERING ---
# We filter the full 52-week forecast down to whatever the user selected on the slider
last_historical_date = historical['Date'].max()
display_forecast = full_forecast[full_forecast['ds'] <= last_historical_date + pd.Timedelta(weeks=forecast_weeks)]

st.subheader(f"Historical Sales vs. {forecast_weeks}-Week AI Forecast")

fig = go.Figure()

# Historical Data
fig.add_trace(go.Scatter(
    x=historical['Date'], y=historical['Weekly_Sales'], mode='lines', name='Historical Sales',
    line=dict(color='#1f77b4', width=2)
))

# Filtered Forecast Data
fig.add_trace(go.Scatter(
    x=display_forecast['ds'], y=display_forecast['yhat'], mode='lines', name='Prophet Forecast',
    line=dict(color='#ff7f0e', width=2, dash='dot')
))

fig.update_layout(xaxis_title="Date", yaxis_title="Total Weekly Sales ($)", hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

st.plotly_chart(fig, use_container_width=True)
st.success(f"✅ Dashboard successfully displaying a {forecast_weeks}-week projection.")