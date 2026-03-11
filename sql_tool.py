import sqlite3

def run_sql_query(sql_query: str):
    try:
        # Connect database created by dataingestion.py
        conn = sqlite3.connect("coffee_sales.db")
        cursor = conn.cursor()
        
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        
        # Get columns
        columns = [desc[0] for desc in cursor.description]
        
        # Convert to disc
        result = [dict(zip(columns, row)) for row in rows]
        
        conn.close()
        
        if not result:
            return "No data was found from the query"
            
        return result[:10]
        
    except Exception as e:
        return f"Error: {e}"

