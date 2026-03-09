import kagglehub
import pandas as pd
from sqlalchemy import create_engine
import os

def fetch_and_store_data():
    print("Steg 1: Hämtar dataset från Kaggle...")
    # Laddar ner senaste versionen av damklädes-datasetet
    path = kagglehub.dataset_download("agungpambudi/trends-product-coffee-shop-sales-revenue-dataset")
    
    # Hittar CSV-filen i den nedladdade mappen
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if not files:
        print("Fel: Hittade ingen CSV-fil.")
        return
    
    csv_path = os.path.join(path, files[0])
    print(f"Steg 2: Läser in {csv_path}")

    # Läser in datan
    df = pd.read_csv(csv_path, sep=',', on_bad_lines='skip')

    if len(df.columns) <= 1:
        df = pd.read_csv(csv_path, sep='|', on_bad_lines='skip')
    
    df = df.loc[:, ~df.columns.str.contains('^unnamed|^ransac', case=False)]
    # RENSNING: Gör kolumnnamnen "Agent-vänliga" (inga mellanslag eller konstiga tecken)
    df.columns = [c.replace(' ', '_').replace('-', '_').replace('/', '_').lower() for c in df.columns]

    df = df.dropna(how='all')
    df['transaction_date']=pd.to_datetime(df['transaction_date'])
    print("Steg 3: Skapar SQLite-databasen 'coffee_sales.db")
    engine = create_engine("sqlite:///coffee_sales.db")
    # Sparar till tabellen 'sales_data'
    df.to_sql("sales_data", engine, if_exists="replace", index=False)
    
    print("\n--- KLART! ---")
    print(f"Totalt antal rader: {len(df)}")
    print("\nKOLUMNER SOM AGENTEN KAN SE:")
    for col in df.columns:
        print(f"- {col}")

if __name__ == "__main__":
    fetch_and_store_data()