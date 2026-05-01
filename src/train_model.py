import pandas as pd
from prophet import Prophet
import os

def train_and_forecast():
    print("Loading clean data...")
    df = pd.read_csv('data/processed/clean_walmart_data.csv')
    
    print("Aggregating sales by date...")
    company_sales = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
    prophet_df = company_sales.rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})
    
    print("Training the upgraded Prophet model...")
    # Add yearly seasonality
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    
    # THE SENIOR UPGRADE: Tell the AI to look for US Holidays (Black Friday, Christmas, etc.)
    model.add_country_holidays(country_name='US')
    
    model.fit(prophet_df)
    
    print("Predicting the next 52 weeks of demand...")
    # Generate a full year of predictions so our Streamlit slider has data to work with
    future = model.make_future_dataframe(periods=52, freq='W')
    forecast = model.predict(future)
    
    print("Saving advanced forecast results...")
    os.makedirs('data/models', exist_ok=True)
    forecast.to_csv('data/models/prophet_forecast.csv', index=False)
    
    print("Success! Advanced AI Model trained.")

if __name__ == "__main__":
    train_and_forecast()