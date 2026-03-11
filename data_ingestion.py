import kagglehub
import pandas as pd
from sqlalchemy import create_engine
import os

def fetch_and_store_data():
    print("Fetching dataset from Kaggle...")
    # Downloading dataset from kaggle
    path = kagglehub.dataset_download("agungpambudi/trends-product-coffee-shop-sales-revenue-dataset")
    
    # Finiding the CSV-file
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if not files:
        print("Fel: Hittade ingen CSV-fil.")
        return
    
    csv_path = os.path.join(path, files[0])
    print(f"Reads {csv_path}")

    # Reads the data
    df = pd.read_csv(csv_path, sep=',', on_bad_lines='skip')

    if len(df.columns) <= 1:
        df = pd.read_csv(csv_path, sep='|', on_bad_lines='skip')
    
    df = df.loc[:, ~df.columns.str.contains('^unnamed|^ransac', case=False)]
    # Cleaning
    df.columns = [c.replace(' ', '_').replace('-', '_').replace('/', '_').lower() for c in df.columns]

    df = df.dropna(how='all')
    df['transaction_date']=pd.to_datetime(df['transaction_date'])
    print("Creates SQLite-database 'coffee_sales.db")
    engine = create_engine("sqlite:///coffee_sales.db")
    # Saves to the table = 'sales_data'
    df.to_sql("sales_data", engine, if_exists="replace", index=False)
    
    print("\n--- FINISHED! ---")
    print(f"Total number of rows: {len(df)}")

if __name__ == "__main__":
    fetch_and_store_data()