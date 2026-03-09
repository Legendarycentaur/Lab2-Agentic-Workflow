from langchain_ollama import ChatOllama
from vector_store import get_schema_advice
from sql_tool import run_sql_query

# 1. Konfigurera "Hjärnan" (Llama 3.1)
# Vi kör temperature=0 för att den ska vara logisk och inte hitta på egna kolumner.
model = ChatOllama(model="llama3.1:8b", temperature=0)

def ask_the_sir(user_question):
    print(f"\n--- [INCOMING QUESTION]: {user_question} ---")

    # STEG 1: RAG - Hämta metadata 
    # Här söker vi i ChromaDB efter vilka kolumner som är relevanta för frågan.
    relevant_info = get_schema_advice(user_question)


    # STEG 2: PLANNER - Skapa SQL (Integration)
    planner_prompt = f"""You are a Supply Chain Optimization Expert.
    Table: 'sales_data'
    Metadata: {relevant_info}

    GOAL: Calculate the OPTIMAL inventory distribution for a specific category based on sales volume.

    STRICT RULES:
    1. Identify the 'product_category' requested.
    2. Calculate the total units sold for that entire category.
    3. Calculate the percentage (%) for each 'product_type' within that category.
    4. Use this SQL structure:
    SELECT 
        product_type, 
        SUM(transaction_qty) as units_sold,
        ROUND(SUM(transaction_qty) * 100.0 / (SELECT SUM(transaction_qty) FROM sales_data WHERE product_category = 'KATEGORI'), 2) as optimal_distribution_pct
    FROM sales_data
    WHERE product_category = 'KATEGORI'
    GROUP BY product_type
    ORDER BY optimal_distribution_pct DESC;

    5. Return ONLY the raw SQL query."""
    
    sql_response = model.invoke([("system", planner_prompt), ("human", user_question)])
    # RENSNING: Säkerställ att vi bara får SQL-koden
    sql_query = sql_response.content.strip().replace("```sql", "").replace("```", "").strip()
    # En extra säkerhetsåtgärd: ta bara det som börjar med SELECT
    if "SELECT" in sql_query.upper():
        sql_query = sql_query[sql_query.upper().find("SELECT"):]
    print(f"-> [SIR'S PLAN]: Executing SQL: {sql_query}")

    # STEG 3: EXECUTION - Hämta data 
    # Vi kör SQL-frågan mot ecommerce_sales.db
    raw_data = run_sql_query(sql_query)
    print(f"-> [DATABASE RESULT]: {raw_data}")

    # STEG 4: SUMMARIZER - Presentera resultatet 
    summarizer_prompt = f"""You are a Senior Inventory Strategist. 
    The user wants to know the OPTIMAL distribution for their stock.
    Data from database: {raw_data}

    TASK: 
    1. Present the percentages as the "Recommended Stock Mix".
    2. Explain that if they were to stock 100 units, they should buy exactly [X] of [Type A], [Y] of [Type B], etc.
    3. Speak in an incredibly posh, authoritative, and helpful manner."""
    
    final_response = model.invoke(summarizer_prompt)
    
    print("\n--- [Final Response] ---")
    print(final_response.content)

if __name__ == "__main__":
    # Denna fråga tvingar nu agenten att agera strategisk rådgivare 
    # och räkna ut fördelningen (distributionen) i procent.
    test_query = "My good Sir, based on our historical sales, what is the optimal percentage distribution of product types for the Coffee category in each of our stores?"
    
    ask_the_sir(test_query)