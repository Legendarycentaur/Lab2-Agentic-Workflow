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
    print(f"Result from RAG:      {relevant_info}")
    print(f"-> [SIR'S MEMORY]: Found relevant columns: {relevant_info.splitlines()[:2]}...")

    # STEG 2: PLANNER - Skapa SQL (Integration)
    planner_prompt = f"""You are a SQLite expert. 
    Table: 'sales_data'
    Metadata: {relevant_info}
    
    IMPORTANT SQLITE RULES:
    1. Use strftime('%m', transaction_date) to get the month.
    2. Use date('now') to get the current date.
    3. The date column is 'transaction_date'.
    4. Return ONLY the raw SQL query, no markdown, no explanation."""
    
    sql_response = model.invoke([("system", planner_prompt), ("human", user_question)])
    sql_query = sql_response.content.strip().replace("```sql", "").replace("```", "").strip()
    print(f"-> [SIR'S PLAN]: Executing SQL: {sql_query}")

    # STEG 3: EXECUTION - Hämta data 
    # Vi kör SQL-frågan mot ecommerce_sales.db
    raw_data = run_sql_query(sql_query)
    print(f"-> [DATABASE RESULT]: {raw_data}")

    # STEG 4: SUMMARIZER - Presentera resultatet 
    summarizer_prompt = f"""You are a fine Sir. You speak in an incredibly posh manner. 
    A user asked: '{user_question}'. 
    The database returned this data: {raw_data}. 
    Present this information to the user with the finest language possible."""
    
    final_response = model.invoke(summarizer_prompt)
    
    print("\n--- [MESSAGE FROM THE SIR] ---")
    print(final_response.content)

if __name__ == "__main__":
    # Prova att köra en fråga!
    ask_the_sir("My good Sir, pray tell, how many coffees were sold last month in all stores?")