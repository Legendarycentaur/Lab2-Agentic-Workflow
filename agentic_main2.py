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
    "GetDatabaseInformation: Provides infromation Abbout the database does not contain data": get_schema_advice,
    "RunSQLQueries: Executes SQL queries corresponding to given infromation": run_sql_query,
    "Calculator: Calulates expressions like (1+1)": safe_calculator
}

# --- ROLL 1: PLANNER (Nu med JSON-säkerhet) ---
def planner_node(goal, history):
    prompt = f"""
    Role: Strategic Professional Problem Solver Planner. 
    Instruction: You understand the Users question perfectly. 
    You know nothing about database unless it is are explicitly given as earlier observations:
    
    Observations:{history}

    available Statuses: CONTINUE, FINNISH
    
    Return ONLY JSON:
    {{ "instruction": "", "status": "CONTINUE" }}"""


    response = model.invoke(f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{prompt}<|eot_id|>
    <|start_header_id|>user<|end_header_id|>{goal}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>""")
    print(response.content)
    return json.loads(response.content)

# --- ROLL 2: CALLER ---
def caller_node(goal, Instruction, history):
    prompt = f"""Role: Technical Tool Utilizer. You follow the instruction precicely you, formulate a query and select a tool, that corresponds to the specific selectedtools requirement. history contains extra context that should be used.
    Instruction: {Instruction} 
    History: {history}
    Tools:{(", ".join(TOOLS.keys()))}
    
    Return ONLY JSON:
    {{ "tool_call": {{ "name": "", "query": "" }}}}"""

    response = model.invoke(f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{prompt}<|eot_id|>
    <|start_header_id|>user<|end_header_id|>{goal}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>""")
    print(response.content)
    return json.loads(response.content)

# --- CONTROLLER LOOP ---
def agent_controller(user_goal):
    state = {"goal": user_goal, "history": [], "retry_count": 0}

    while state["retry_count"] < 10:
        # Hämta plan
        plan = planner_node(state["goal"], state["history"][-3:])
        
        # FIX: Kontrollera att plan faktiskt innehåller 'instruction' (löser ditt TypeError)
        instruction = plan.get("instruction")
        print(f"-> [PLANNER]: {instruction}")

        if plan.get("status") == "FINISH":
            # Här anropar vi en summarizer eller returnerar sista observationen
            return "Analysis complete. Distribution: " + str(state["history"][-1]["observation"])

        # Hämta tool call
        call_data = caller_node(state["goal"], instruction, state["history"])
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
                    "action": t_name,
                    "earlierObservations": observation
                })
        
        state["retry_count"] += 1

    return "The Sir is exhausted."

if __name__ == "__main__":
    test_query = "My good Sir, calculate the percentage distribution of product types for the Coffee category."
    print(f"\n--- [SIR'S AUDITED REPORT] ---\n{agent_controller(test_query)}")