import json
import re
from langchain_ollama import ChatOllama
from vector_store import get_schema_advice
from sql_tool import run_sql_query

#CONFIGURATION
model = ChatOllama(model="llama3.1:8b", temperature=0, format="json")

def extract_json(response_text):
    try:
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(response_text)
    except Exception:
        return None

#TOOLs
def safe_calculator(expression: str) -> str:
    if not expression or expression == "null": return "ERROR: Missing expression."
    
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

# --- ROLL 1: PLANNER---
def planner_node(goal, full_history):
    history_str = ""
    has_schema_info = False
    has_sql_data = False
    total_sum_found = ""
    percentages_result = ""
    has_validation = False
    has_percentages = False
    
    for i, h in enumerate(full_history):
        obs = str(h.get('observation', ''))
        action = h.get('action', '')
        query = str(h.get('query', ''))
        
        history_str += f"Step {i+1}: {action}({query}) -> {obs[:1000]}\n"
        
        if action == "get_schema": has_schema_info = True
        if action == "run_sql" and "SUM" in query.upper() and "Error" not in obs and "error" not in obs.lower(): 
            has_sql_data = True
        if action == "calculator" and "+" in query and "Err" not in obs and "ERROR" not in obs: 
            if not has_percentages: total_sum_found = obs 
        if action == "calculator" and "/" in query and "Err" not in obs and "ERROR" not in obs: 
            has_percentages = True
            percentages_result = obs
        if action == "calculator" and has_percentages and "+" in query and "Err" not in obs and "ERROR" not in obs:
            if "99." in obs or "100" in obs or "101." in obs:
                has_validation = True

    # --- DETERMINE SUGGESTED NEXT STEP ---
    if has_validation:
        suggested_instruction = "FINISH"
        suggested_thought = "Validation successful. Percentages sum to 100%."
    elif has_percentages:
        parts = percentages_result.split(",")
        sum_query = " + ".join([p.strip() for p in parts if p.strip()])
        suggested_instruction = f"CALCULATOR: {sum_query}"
        suggested_thought = "Validate that the calculated percentages sum to exactly 100."
    elif total_sum_found:
        suggested_instruction = f"CALCULATOR: [val1/{total_sum_found}*100, val2/{total_sum_found}*100, val3/{total_sum_found}*100] (Replace val1, val2 etc. with EXACT quantities from the SQL result IN THEIR ORIGINAL ORDER)"
        suggested_thought = f"Total sum is {total_sum_found}. I MUST use the CALCULATOR tool to get the exact percentages. I will NOT guess or calculate them myself."
    elif has_sql_data:
        suggested_instruction = "CALCULATOR: <num1> + <num2> + <num3>... (REPLACE with actual quantities from SQL result. NO SQL QUERIES. USE ONLY NUMBERS, NOT TEXT)"
        suggested_thought = "Sum the quantities using the calculator with REAL NUMBERS."
    elif has_schema_info:
        suggested_instruction = "SQL: SELECT product_type, SUM(transaction_qty) FROM sales_data WHERE product_category = '' AND transaction_date LIKE '<YYYY-MM>%' AND store_location = '<location>' GROUP BY product_type"
        suggested_thought = "Write a SQL query to get the total quantity per product type."
    else:
        suggested_instruction = "SCHEMA: sales_data"
        suggested_thought = "Start by fetching the table schema."

    prompt = f"""Role: Strategic Planner. Goal: {goal}.
    
    HISTORY OF ACTIONS: 
    {history_str if history_str else "Starting now."}
    
    SUGGESTED NEXT STEP:
    Thought: {suggested_thought}
    Instruction: {suggested_instruction}
    
    INSTRUCTIONS:
    Review the history. If you made a mistake or hit an error, correct your course.
    Otherwise, follow the SUGGESTED NEXT STEP. Remember, the CALCULATOR tool ONLY accepts real numbers (e.g., 24943/38727*100), NEVER text variables or column names.
    
    Return ONLY JSON matching this structure:
    {{
        "analysis": "Evaluate history. Do we follow the suggestion or fix an error?",
        "thought": "Your intent for this turn.",
        "instruction": "The command starting with SCHEMA:, SQL:, CALCULATOR:, or FINISH.", 
        "status": "FINISH if instruction is FINISH, else CONTINUE"
    }}"""
    
    res = model.invoke(prompt)
    return extract_json(res.content) or {"instruction": "SCHEMA: sales_data", "status": "CONTINUE"}

# --- ROLL 2: CALLER ---
def caller_node(instruction, goal): #Goal används inte
    prompt = f"""Role: Technical Assistant.
    Instruction from Planner: {instruction}
    
    YOUR JOB: Extract the exact query based on the prefix.
    - If prefix is 'SCHEMA:', tool is 'get_schema'. Query is the text after the prefix.
    - If prefix is 'SQL:', tool is 'run_sql'. Query is the pure SQL statement.
    - If prefix is 'CALCULATOR:', tool is 'calculator'. NO SQL QUERY, Math Query MUST BE ONLY NUMBERS AND OPERATORS (+, -, /, *). REMOVE ALL WORDS.
    
    Return ONLY JSON:
    {{
        "tool_call": {{
            "name": "calculator" or "run_sql" or "get_schema",
            "query": "exact string here"
        }}
    }}"""
    
    res = model.invoke(prompt)
    data = extract_json(res.content)
    
    # if toolcall can not be parsed
    if not data or "tool_call" not in data:
        return {"tool_call": {"name": "calculator", "query": "ERROR_JSON_PARSE_FAILED"}}
        
    
    return data

# --- ROLL 3: SUMMARIZER  ---
def summarizer_node(goal, history):
    prompt = f"""Role: Posh product planning Strategist. Goal: {goal}. History: {json.dumps(history)}. 
    Write a concise, formal final report mapping the calculated percentages to the correct coffee types.
    You can happily use a table. 
    Return JSON format where final report is a text string with what you have to say: {{'final_report': '...'}}
    """

    res = extract_json(model.invoke(prompt).content).get("final_report", "Complete.")
    
    prompt = f"""Role: Posh product planning Strategist. Goal: {goal}. History: {json.dumps(history)}. 
    Use the History and describe every thought and its result -> describe it in your words.
    Do this in a unordered list format.  
    Return JSON format where final report is a text string with what you have to say: {{'final_report': '...'}}"""
    res += "\n\n" + extract_json(model.invoke(prompt).content).get("final_report", "Complete.")
    return res

# --- CONTROLLER ---
def agent_controller(user_goal):
    state = {"goal": user_goal, "history": [], "step_count": 0}
    last_action = None

    print("\n--- [STARTING AGENTIC FLOW] ---")

    while state["step_count"] < 20:
        plan = planner_node(state["goal"], state["history"])
        
        if plan.get("status") == "FINISH":
            print("\n[SYSTEM]: Goal reached. Finalizing...")
            return summarizer_node(state["goal"], state["history"])

        call = caller_node(plan.get("instruction"), state["goal"])
        tool_info = call.get("tool_call", {})
        t_name = tool_info.get("name")
        t_query = str(tool_info.get("query", ""))

        current_action = f"{t_name}:{t_query}"
        if current_action == last_action:
            print(f"\n!! [GUARD BLOCK]: Agent is repeating '{current_action}'. Forcing transition.")
            state["history"].append({"action": "system", "observation": "ERROR: You repeated the exact same query. Advance to the next logic step!"})
            state["step_count"] += 1
            continue
            
        last_action = current_action

        print(f"\n[STEP {state['step_count']+1}] {t_name}: {t_query}")

        if t_name in TOOLS:
            try:
                obs = str(TOOLS[t_name](t_query))
                print(f"Observation: {obs[:150]}...")
                state["history"].append({"action": t_name, "query": t_query, "observation": obs,"thought": plan.get("thought")})
            except Exception as e:
                state["history"].append({"action": t_name, "query": t_query, "observation": f"Error: {e}","thought": plan.get("thought")})

        state["step_count"] += 1
    result = summarizer_node(state["goal"], state["history"])
    return "Failed to determine a n answer to the question."

if __name__ == "__main__":
    print(agent_controller("Calculate the percentage distribution of product types for the tea category in lower manhattan last month. Time is 2023-03"))