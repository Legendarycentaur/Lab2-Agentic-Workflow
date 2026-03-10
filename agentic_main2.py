import json
import re
from langchain_ollama import ChatOllama
from vector_store import get_schema_advice
from sql_tool import run_sql_query

# 1. KONFIGURATION
model = ChatOllama(model="llama3.1:8b", temperature=0, format="json")

def extract_json(response_text):
    try:
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(response_text)
    except Exception:
        return None

# 2. VERKTYG (TOOLS)
def safe_calculator(expression: str) -> str:
    """Hanterar beräkningar. Tar nu brutalt bort ALLA bokstäver från inputen."""
    if not expression or expression == "null": return "ERROR: Missing expression."
    
    # Ta bort allt som är bokstäver (förhindrar alla SQL-försök)
    clean_str = re.sub(r"[a-zA-Z]", "", str(expression))
    expressions = clean_str.split(',')
    results = []
    
    for expr in expressions:
        clean_expr = re.sub(r"[^0-9\.\+\-\*\/\(\)\ ]", "", expr).strip()
        clean_expr = re.sub(r"\++", "+", clean_expr).strip("+")
        if clean_expr:
            try:
                res = round(eval(clean_expr, {"__builtins__": {}}, {}), 2)
                results.append(str(res))
            except:
                results.append("Err")
                
    if not results:
        return "ERROR: No valid math found."
    return ", ".join(results)

TOOLS = {
    "get_schema": get_schema_advice,
    "run_sql": run_sql_query,
    "calculator": safe_calculator
}

# --- ROLL 1: PLANNER (Den analytiska strategen) ---
def planner_node(goal, full_history):
    history_str = ""
    
    has_schema_info = False
    has_sql_data = False
    total_sum_found = ""
    has_percentages = False
    
    for i, h in enumerate(full_history):
        obs = str(h.get('observation', ''))
        action = h.get('action', '')
        query = str(h.get('query', ''))
        
        history_str += f"Step {i+1}: {action} -> {obs[:1000]}\n"
        
        if action == "get_schema": has_schema_info = True
        if action == "run_sql" and "SUM" in obs and "product_type" in obs: has_sql_data = True
        if action == "calculator" and "+" in query and "Err" not in obs and "ERROR" not in obs: total_sum_found = obs 
        if action == "calculator" and "/" in query and "Err" not in obs and "ERROR" not in obs: has_percentages = True

    # --- PREFIX-STYRD LOGIK ---
    if has_percentages:
        cmd = "FINISH"
        thought = "PHASE 4: Percentages successfully calculated!"
    elif total_sum_found:
        cmd = f"CALCULATOR: Divide each product's quantity from the SQL result by {total_sum_found} and multiply by 100. Write ONLY the equations separated by commas. Example: 24943/{total_sum_found}*100, 12891/{total_sum_found}*100"
        thought = f"PHASE 3: I have the total sum ({total_sum_found}). I will output a CALCULATOR command with the division equations."
    elif has_sql_data:
        cmd = "CALCULATOR: Extract the numbers from the SQL result and add them using '+'. Example: 24943 + 12891 + 25973 + 13012 + 12431"
        thought = "PHASE 2: I must sum the quantities. I will output a CALCULATOR command."
    elif has_schema_info:
        cmd = "SQL: SELECT product_type, SUM(transaction_qty) FROM sales_data WHERE product_category = 'Coffee' GROUP BY product_type"
        thought = "PHASE 1: I know the schema. I will output a SQL command."
    else:
        cmd = "SCHEMA: sales_data"
        thought = "PHASE 0: I must fetch the schema first."

    prompt = f"""Role: Strategic Planner. Goal: {goal}.
    HISTORY: 
    {history_str if history_str else "Starting now."}
    
    INSTRUCTION TO SEND: {cmd}
    
    Return ONLY JSON:
    {{
        "thought": "{thought}",
        "instruction": "{cmd}", 
        "status": "{'FINISH' if cmd == 'FINISH' else 'CONTINUE'}"
    }}"""
    
    res = model.invoke(prompt)
    return extract_json(res.content) or {"instruction": "SCHEMA: sales_data", "status": "CONTINUE"}

# --- ROLL 2: CALLER (Teknikern) ---
def caller_node(instruction, goal):
    prompt = f"""Role: Technical Assistant.
    Instruction from Planner: {instruction}
    
    YOUR JOB: Extract the exact query based on the prefix.
    - If prefix is 'SCHEMA:', tool is 'get_schema'. Query is the text after the prefix.
    - If prefix is 'SQL:', tool is 'run_sql'. Query is the pure SQL statement.
    - If prefix is 'CALCULATOR:', tool is 'calculator'. Query MUST BE ONLY NUMBERS AND OPERATORS (+, -, /, *). REMOVE ALL WORDS.
    
    Return ONLY JSON:
    {{
        "tool_call": {{
            "name": "calculator" or "run_sql" or "get_schema",
            "query": "exact string here"
        }}
    }}"""
    
    res = model.invoke(prompt)
    data = extract_json(res.content)
    
    # Om tolkningen misslyckas tvingar vi fram ett system-fel istället för att dölja det
    if not data or "tool_call" not in data:
        return {"tool_call": {"name": "calculator", "query": "ERROR_JSON_PARSE_FAILED"}}
        
    # Säkerställ mappning baserat på prefixet ifall agenten försöker vara kreativ
    t = data["tool_call"]
    if "SCHEMA:" in instruction: t["name"] = "get_schema"
    elif "SQL:" in instruction: t["name"] = "run_sql"
    elif "CALCULATOR:" in instruction: t["name"] = "calculator"
    
    return data

# --- ROLL 3: SUMMARIZER (The Sir) ---
def summarizer_node(goal, history):
    prompt = f"Role: Posh Strategist. Goal: {goal}. History: {json.dumps(history)}. Write a concise, formal final report mapping the calculated percentages to the correct coffee types. Return JSON: {{'final_report': '...'}}"
    res = model.invoke(prompt)
    return extract_json(res.content) or {"final_report": "Report failed."}

# --- CONTROLLER (Motorn) ---
def agent_controller(user_goal):
    state = {"goal": user_goal, "history": [], "retry_count": 0}
    last_action = None

    print("\n--- [STARTING AGENTIC FLOW] ---")

    while state["retry_count"] < 12:
        plan = planner_node(state["goal"], state["history"])
        
        if plan.get("status") == "FINISH":
            print("\n[SYSTEM]: Goal reached. Finalizing...")
            return summarizer_node(state["goal"], state["history"]).get("final_report", "Complete.")

        call = caller_node(plan.get("instruction"), state["goal"])
        tool_info = call.get("tool_call", {})
        t_name = tool_info.get("name")
        t_query = str(tool_info.get("query", ""))

        current_action = f"{t_name}:{t_query}"
        if current_action == last_action:
            print(f"\n!! [GUARD BLOCK]: Agent is repeating '{current_action}'. Forcing transition.")
            state["history"].append({"action": "system", "observation": "ERROR: You repeated the exact same query. Advance to the next logic step!"})
            state["retry_count"] += 1
            continue
            
        last_action = current_action

        print(f"\n[STEP {state['retry_count']+1}] {t_name}: {t_query}")

        if t_name in TOOLS:
            try:
                obs = str(TOOLS[t_name](t_query))
                print(f"Observation: {obs[:150]}...")
                state["history"].append({"action": t_name, "query": t_query, "observation": obs})
            except Exception as e:
                state["history"].append({"action": t_name, "query": t_query, "observation": f"Error: {e}"})

        state["retry_count"] += 1

    return "The Sir is exhausted."

if __name__ == "__main__":
    print(agent_controller("Calculate the percentage distribution of product types for the Coffee category."))