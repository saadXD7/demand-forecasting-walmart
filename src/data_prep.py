import pandas as pd
import os

def load_and_merge_data():
    print("Loading raw data...")
    # Read the three CSV files (paths assume you run this from the main folder)
    train = pd.read_csv('data/raw/train.csv')
    features = pd.read_csv('data/raw/features.csv')
    stores = pd.read_csv('data/raw/stores.csv')

    print("Merging datasets...")
    # 1. Add store details (like Store Type and Size) to the sales data
    df = pd.merge(train, stores, on='Store', how='left')
    
    # 2. Add features (like Temperature, Fuel Price, and Markdowns)
    df = pd.merge(df, features, on=['Store', 'Date', 'IsHoliday'], how='left')

    print("Cleaning dates and handling missing values...")
    # Convert Date string to actual Datetime object (Crucial for Time Series)
    df['Date'] = pd.to_datetime(df['Date'])

    # Handle Missing Data: If a Markdown is NaN, it means there was no discount that day
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    df[markdown_cols] = df[markdown_cols].fillna(0)

    print("Engineering Time Features...")
    # Break the date down so the model can understand seasonality
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week

    return df

if __name__ == "__main__":
    # Ensure the processed folder exists
    os.makedirs('data/processed', exist_ok=True)

    # Execute the function
    final_df = load_and_merge_data()

    # Save the clean, merged data
    output_path = 'data/processed/clean_walmart_data.csv'
    final_df.to_csv(output_path, index=False)
    print(f"Success! Clean data saved to {output_path} with {len(final_df)} rows.")