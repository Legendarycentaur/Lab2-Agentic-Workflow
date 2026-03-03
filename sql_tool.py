import sqlite3

def run_sql_query(sql_query: str):
    """
    Kör en SQL-fråga mot ecommerce_sales.db och returnerar resultatet.
    Används av agenten för att hämta försäljningsdata.
    """
    try:
        # Anslut till databasen som skapades av dataingestion.py
        conn = sqlite3.connect("ecommerce_sales.db")
        cursor = conn.cursor()
        
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        
        # Hämta kolumnnamn så att LLM:en förstår vad värdena betyder
        columns = [desc[0] for desc in cursor.description]
        
        # Omvandla till en lista med ordböcker (dicts) för bättre läsbarhet för AI:n
        result = [dict(zip(columns, row)) for row in rows]
        
        conn.close()
        
        if not result:
            return "Ingen data hittades för den frågan."
            
        # Vi returnerar max 10 rader för att inte skicka för mycket text till modellen
        return result[:10]
        
    except Exception as e:
        return f"Fel vid körning av SQL: {e}"

# Ett litet testblock för att se att kopplingen fungerar direkt
if __name__ == "__main__":
    print("Testar koppling till databasen...")
    test_result = run_sql_query("SELECT size, SUM(quantity) as total_sold FROM sales_data GROUP BY size ORDER BY total_sold DESC LIMIT 5")
    print(test_result)