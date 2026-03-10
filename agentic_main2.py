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
    """Uppgraderad kalkylator som klarar av flera beräkningar samtidigt!"""
    clean_str = str(expression).lower().replace("sum", "").replace("(", "").replace(")", "")
    
    # Dela upp vid kommatecken för att hantera flera uttryck
    expressions = clean_str.split(',')
    results = []
    
    for expr in expressions:
        # Rensa varje separat uttryck
        clean_expr = re.sub(r"[^0-9\.\+\-\*\/\ ]", "", expr).strip()
        clean_expr = re.sub(r"\++", "+", clean_expr).strip("+")
        
        if clean_expr:
            try:
                res = round(eval(clean_expr, {"__builtins__": {}}, {}), 2)
                results.append(str(res))
            except Exception as e:
                results.append(f"Err({clean_expr})")
                
    if not results:
        return "ERROR: No valid math found."
    
    # Returnera en lista med resultat, t.ex. "30.29, 15.65, 54.06"
    return ", ".join(results)

TOOLS = {
    "get_schema": get_schema_advice,
    "run_sql": run_sql_query,
    "calculator": safe_calculator
}

# --- ROLL 1: PLANNER (Tillståndsmaskinen) ---
def planner_node(goal, full_history):
    history_str = ""
    
    # Fas-indikatorer
    has_sql_data = False
    total_sum_found = ""
    has_percentages = False
    
    for i, h in enumerate(full_history):
        obs = str(h.get('observation', ''))
        action = h.get('action', '')
        query = str(h.get('query', ''))
        
        history_str += f"Step {i+1}: {action} -> {obs[:1000]}\n"
        
        if action == "run_sql" and "count" in obs:
            has_sql_data = True
        if action == "calculator" and "+" in query and not "Err" in obs:
            total_sum_found = obs # Sparar resultatet av additionen
        if action == "calculator" and "/" in query:
            has_percentages = True

    # --- TILLSTÅNDSLOGIK (STATE MACHINE) ---
    if has_percentages:
        instruction_nudge = "PHASE 3 COMPLETE: Percentages calculated! Your ONLY task now is to set status to 'FINISH'."
    elif total_sum_found:
        instruction_nudge = f"PHASE 2: TOTAL SUM IS {total_sum_found}. Now calculate the percentages for each product. Instruction should be parts divided by {total_sum_found} multiplied by 100. Example instruction: 'Calculate 16403/{total_sum_found}*100, 8477/{total_sum_found}*100'"
    elif has_sql_data:
        instruction_nudge = "PHASE 1: DATA DETECTED. First, we need the total sum. Write an instruction to ADD all the counts together. Example instruction: 'Sum 16403 + 8477 + 16912'"
    else:
        instruction_nudge = "PHASE 0: Fetch data. Use: SELECT product_type, COUNT(*) as count FROM sales_data WHERE product_category = 'Coffee' GROUP BY product_type"

    prompt = f"""Role: Strategic Planner. Goal: {goal}.
    
    HISTORY:
    {history_str if history_str else "Starting now."}

    GUIDELINE:
    {instruction_nudge}

    Return ONLY JSON:
    {{
        "thought": "Brief reasoning based on the current Phase",
        "instruction": "Specific command to the technician", 
        "status": "CONTINUE or FINISH"
    }}"""
    
    res = model.invoke(prompt)
    return extract_json(res.content) or {"instruction": "get_schema", "status": "CONTINUE"}

# --- ROLL 2: CALLER (Teknikern) ---
def caller_node(instruction, goal):
    prompt = f"""Role: Technical Assistant. 
    Goal: {goal}.
    Instruction: {instruction}.
    
    RULES:
    1. If instruction is to get data -> tool: 'run_sql', query: 'SELECT ...'
    2. If instruction has numbers to add/divide/calculate -> tool: 'calculator', query: ONLY NUMBERS AND OPERATORS. Comma separated is allowed!
    
    EXAMPLES OF CORRECT CALCULATOR USE:
    Instruction: "Sum 16403 + 8477"
    Return: {{ "tool_call": {{ "name": "calculator", "query": "16403 + 8477" }} }}
    
    Instruction: "Calculate 16403/50000*100, 8477/50000*100"
    Return: {{ "tool_call": {{ "name": "calculator", "query": "16403/50000*100, 8477/50000*100" }} }}
    
    Return ONLY JSON:"""
    
    res = model.invoke(prompt)
    data = extract_json(res.content)
    
    t = data.get("tool_call", {}) if data else {}
    name = str(t.get("name")).lower()
    if any(x in name for x in ["calc", "math", "sum"]): t["name"] = "calculator"
    elif "sql" in name: t["name"] = "run_sql"
    
    return data or {"tool_call": {"name": "get_schema", "query": "sales_data"}}

# --- ROLL 3: SUMMARIZER (The Sir) ---
def summarizer_node(goal, history):
    prompt = f"Role: Posh Strategist. Goal: {goal}. History: {json.dumps(history)}. Write a formal report stating the final percentage distribution of the coffee types. Return JSON: {{'final_report': '...'}}"
    res = model.invoke(prompt)
    return extract_json(res.content) or {"final_report": "Report generation failed."}

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
        t_query = tool_info.get("query")

        # Säkerhetsspärrar
        if t_name == "calculator" and "select" in str(t_query).lower():
            state["history"].append({"action": "system", "observation": "ERROR: Calculator only takes math, not SQL."})
            state["retry_count"] += 1
            continue

        current_action = f"{t_name}:{t_query}"
        if current_action == last_action:
            state["history"].append({"action": "system", "observation": "ERROR: Repeating action. Advance to the next phase!"})
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

    return "The Sir is exhausted. Coordination failure."

if __name__ == "__main__":
    print(agent_controller("Calculate the percentage distribution of product types for the Coffee category."))