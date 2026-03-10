import json
import re
from langchain_ollama import ChatOllama
from vector_store import get_schema_advice # Här används dina Embeddings/Vector Store
from sql_tool import run_sql_query

# 1. Konfiguration
model = ChatOllama(model="llama3.1:8b", temperature=0, format="json")

def safe_calculator(x):
    try:
        # Vi tillåter bara siffror och operatorer för att undvika SQL-injektion i eval
        if any(word in x.upper() for word in ["SELECT", "FROM", "WHERE", "UPDATE"]):
            return "Error: Calculator received SQL instead of math. Use run_sql for database queries."
        return str(eval(x, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Math Error: {str(e)}. Please provide a simple numeric expression like '100 / 500 * 100'."

TOOLS = {
    "get_schema": get_schema_advice,
    "run_sql": run_sql_query,
    "calculator": safe_calculator
}

# --- ROLL 1: PLANNER (Nu med JSON-säkerhet) ---
def planner_node(goal, history):
    prompt = f"""Role: Strategic Planner. Goal: {goal}. History: {history}.
    
    MY CHECKLIST:
    1. Do I know the column names? If NO -> Next step: "get_schema"
    2. Do I have the raw sales numbers? If NO -> Next step: "run_sql"
    3. Do I have the numbers but need percentages? If YES -> Next step: "calculator"
    4. Are all percentages done? If YES -> status: "FINISH"
    
    Return ONLY JSON:
    {{ "instruction": "Next step from checklist", "status": "CONTINUE" }}"""
    
    response = model.invoke(prompt)
    return json.loads(response.content)

# --- ROLL 2: CALLER ---
def caller_node(instruction, history):
    prompt = f"""Role: Technical Tool Selector. 
    Instruction: {instruction}. 
    
    RULES FOR TOOLS:
    - If instruction is to find columns/tables -> name: "get_schema", query: "sales"
    - If instruction is to get data from DB -> name: "run_sql", query: "SELECT..."
    - If instruction is to do MATH with numbers -> name: "calculator", query: "numbers only (e.g. 10+10)"
    
    Return ONLY JSON:
    {{ "tool_call": {{ "name": "get_schema" or "run_sql" or "calculator", "query": "string" }} }}"""
    
    response = model.invoke(prompt)
    return json.loads(response.content)

# --- CONTROLLER LOOP ---
def agent_controller(user_goal):
    state = {"goal": user_goal, "history": [], "retry_count": 0}

    while state["retry_count"] < 10:
        # Hämta plan
        plan = planner_node(state["goal"], state["history"][-4:])
        
        # FIX: Kontrollera att plan faktiskt innehåller 'instruction' (löser ditt TypeError)
        instruction = plan.get("instruction", "get_schema")
        print(f"-> [PLANNER]: {instruction}")

        if plan.get("status") == "FINISH":
            # Här anropar vi en summarizer eller returnerar sista observationen
            return "Analysis complete. Distribution: " + str(state["history"][-1]["observation"])

        # Hämta tool call
        call_data = caller_node(instruction, state["history"])
        tool_info = call_data.get("tool_call")
        
        # Hantera om tool_info är en lista (vanligt fel hos 8B)
        if isinstance(tool_info, list): tool_info = tool_info[0]

        if tool_info:
            t_name = tool_info.get("name")
            t_query = tool_info.get("query")
            print(f"-> [CALLER]: Selected {t_name} with query {t_query}")
            if t_name in TOOLS:
                print(f"-> [EXECUTING]: {t_name}")
                # HÄR anropas din Vector Store via get_schema
                observation = str(TOOLS[t_name](t_query))
                
                state["history"].append({
                    "plan": instruction,
                    "action": t_name,
                    "query": t_query,
                    "observation": observation
                })
        
        state["retry_count"] += 1

    return "The Sir is exhausted."

if __name__ == "__main__":
    test_query = "My good Sir, calculate the percentage distribution of product types for the Coffee category."
    print(f"\n--- [SIR'S AUDITED REPORT] ---\n{agent_controller(test_query)}")