from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
from vector_store import get_schema_advice
from sql_tool import run_sql_query
import json

# 1. Konfigurera "Hjärnan" (Llama 3.1)
# Vi kör temperature=0 för att den ska vara logisk och inte hitta på egna kolumner.
model = ChatOllama(model="llama3.1:8b", temperature=0)

def ask_the_sir(user_question):
    print(f"\n--- [INCOMING QUESTION]: {user_question} ---")

    # STEG 1: RAG - Hämta metadata 
    # Här söker vi i ChromaDB efter vilka kolumner som är relevanta för frågan.
    relevant_info = get_schema_advice(f"search_query:{user_question}")
    print(f"Result from RAG:      {relevant_info}")
    print(f"-> [SIR'S MEMORY]: Found relevant columns: {relevant_info.splitlines()[:2]}...")

    # STEG 2: PLANNER - Skapa SQL (Integration)
    planner_prompt = f"""You are a SQLite expert. 
    Table: 'sales_data'
    Metadata: {relevant_info}
    
    IMPORTANT SQLITE RULES:
    1. Use date('now') to get the current date if needed. 
    2. strftime('%m', columnname) is used to get the month. 
    3. The date is very important do not f-up the dates  
    4. Return ONLY the raw SQL query, no markdown, no explanation.
    5. Never hallucinate
    
    use this format but replace with real variable names: 
    WITH MonthlySales AS (
    SELECT 
        product_name, 
        COUNT(*) as units_sold
    FROM 
        sales
    WHERE 
        category = ''
        sale_date >= date()
        AND sale_date < date()
    GROUP BY 
        product_name
),
TotalCount AS (
    SELECT SUM(units_sold) as grand_total FROM MonthlySales
)
SELECT 
    product_name,
    units_sold,
    ROUND((units_sold * 100.0) / (SELECT grand_total FROM TotalCount), 2) || '%' AS percentage_of_total
FROM 
    MonthlySales
ORDER BY 
    units_sold DESC;"""
    
    sql_response = model.invoke([("system", planner_prompt), ("human", user_question)])
    sql_query = sql_response.content.strip().replace("```sql", "").replace("```", "").strip()
    print(f"-> [SIR'S PLAN]: Executing SQL: {sql_query}")

    # STEG 3: EXECUTION - Hämta data 
    # Vi kör SQL-frågan mot ecommerce_sales.db
    raw_data = run_sql_query(sql_query)
    print(f"-> [DATABASE RESULT]: {raw_data}")

    # STEG 4: SUMMARIZER - Presentera resultatet 
    summarizer_prompt = f""" 
    The database returned this data: {raw_data}. 
    Present this information to the user in the json format nothing else. AND Do not say the format "json" It should contain the database information. The property_name should be enclosed in " """
    
    json_response = model.invoke(summarizer_prompt)
    parser = JsonOutputParser()
    # json_parsed_response = parser.parse(text=json_response.content)
    print(json_response.content)
    json_parsed_parsed_response = json.loads(json_response.content)

    # print(json)
    print("\n--- [MESSAGE FROM THE SIR] ---")
    # print(json_response.content)
    total = 0;
    for line in json_parsed_parsed_response:
        total += line.get("percentage_of_total") 
        print(line.get("percentage_of_total"))
    

if __name__ == "__main__":
    # Prova att köra en fråga!
    ask_the_sir("My good Sir, pray tell, todays date is: 2023-03-01!!, What is the distribution of sales last month for the category tea?")