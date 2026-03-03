import kagglehub
import pandas as pd
from sqlalchemy import create_engine
import os

def fetch_and_store_data():
    print("Steg 1: Hämtar dataset från Kaggle...")
    # Laddar ner senaste versionen av damklädes-datasetet
    path = kagglehub.dataset_download("shilongzhuang/-women-clothing-ecommerce-sales-data")
    
    # Hittar CSV-filen i den nedladdade mappen
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if not files:
        print("Fel: Hittade ingen CSV-fil.")
        return
    
    csv_path = os.path.join(path, files[0])
    print(f"Steg 2: Läser in {csv_path}")

    # Läser in datan
    df = pd.read_csv(csv_path)
    
    # RENSNING: Gör kolumnnamnen "Agent-vänliga" (inga mellanslag eller konstiga tecken)
    df.columns = [c.replace(' ', '_').replace('-', '_').replace('/', '_').lower() for c in df.columns]

    print("Steg 3: Skapar SQLite-databasen 'ecommerce_sales.db'...")
    engine = create_engine("sqlite:///ecommerce_sales.db")
    
    # Sparar till tabellen 'sales_data'
    df.to_sql("sales_data", engine, if_exists="replace", index=False)
    
    print("\n--- KLART! ---")
    print(f"Totalt antal rader: {len(df)}")
    print("\nKOLUMNER SOM AGENTEN KAN SE:")
    for col in df.columns:
        print(f"- {col}")

if __name__ == "__main__":
    fetch_and_store_data()