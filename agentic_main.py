import json
from typing import Any, Dict, List, Literal

from langchain_ollama import ChatOllama
from vector_store import get_schema_advice
from sql_tool import run_sql_query
from StatePrompt import build_state_prompt
from langchain_community.utilities import WolframAlphaAPIWrapper
from calculator import calculator
# 1. Konfigurera "Hjärnan" (Llama 3.1)
# Vi kör temperature=0 för att den ska vara logisk och inte hitta på egna kolumner.

# def jsonReformat(planerResult):
#     return json.loads(plannerResult.content.strip())

Plan:json

def UseTool(model:ChatOllama, planner_question, plannerTool):
    print("tool")
    return caller(model,planner_question,plannerTool)


def Replan(self, model:ChatOllama, callers_question):
    self.PlannerPrompt = "You are an awsome planner, You will update the provided plan to fix the callers question. You ONLY return in the same JSON FORMAT. Nothing else. " \
    f"CurrentPLAN: {self.Plan}"
    planner(model=model, planner_question=callers_question)
    print("plan")

def createPlanQuestion(str):
    print("replan question")


def Finnish(model:ChatOllama, planner_question, plannerTool):
    print("finnish")
    return (0,0)

def planner(self, model:ChatOllama, planner_question):
    plannerResult = model.invoke(f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{PlannerPrompt}<|eot_id|>
    <|start_header_id|>user<|end_header_id|>{planner_question}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>""")
    self.Plan = json.loads(plannerResult.content.strip())

def planning(model:ChatOllama, planner_question, plannerTool):
    print("used planning")
    return (0,0)


TOOLS = {
    "GetDatabaseColumns: Embedding model that  give information about the database and its tables it wants a question. ": get_schema_advice,
    "RunSQLQuery: Executes SQL queries to the database, needs input formulated as VALID SQL": run_sql_query,
    "Calculator: Calculates mathematical expressions, needs input as formulated equation like (1+1) NO SQL": calculator,
    "Plan: Updates the plan to answer question given": createPlanQuestion
    }

ActionTypes = {
    "Use Tool: Creates tool with instruction, Can be empty":UseTool,
    "Plan: This ends the session and creates final answer to user":planning,
    "Finnish: This ends the session and creates final answer to user":Finnish
}

PlannerPrompt = "You are a an awsome planner:" \
"You understand exactly what the user wants. " \
f"Tools: {", ".join(TOOLS.keys())} \n" \
f"ActionTypes: {", ".join(ActionTypes)} \n" \
"You return a list of steps in JSON format. You do not know ANYTHING about the columns. " \
"The steps represent what tools and actions to do to get result" \
"""The query should be in json format it will be used for a different llm to understand use this format: 
[
{
  "step":0,
  "ActionType": ""
  "toolType":"",
  "context": ""
},
]""" \
"You do not add any SQL queries in the parameters.  You give instruction for llm to understand how to build sql query."\
"""You only return the JSON format nothing else.
When you have the final answer, use the tool: "Finish" with the input being the final answer. """



def caller(model:ChatOllama, planner_question:str, PlannerTool):
    
    CallerInstruction = f"Tools: {", ".join(TOOLS.keys())}" \
    f"The planner decided you should use: {PlannerTool}" \
    """Your job is to decide what tool to use. You do not know anything about the databse columns: These should have been provided. If not call Plan. And what to send to it. You return only a json in this format: {"tool":"","input":""} nothing else.""" 
    
    callerResult = model.invoke(f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{CallerInstruction}<|eot_id|>
    <|start_header_id|>user<|end_header_id|>{planner_question}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>""")

    print("CallerContent: "+ callerResult.content)
    
    cleaned_content = callerResult.content.strip().replace("```json", "").replace("```", "").strip()

    try:
        caller_data = json.loads(cleaned_content)
    except json.JSONDecodeError:
        print("Error: Invalid JSON returned by the model.")
        # Return a specific feedback message as the observation so the agent knows to fix it
        error_feedback = "Error: Your previous response was not valid JSON. Please provide ONLY a valid JSON object without any additional text or markdown."
        return (error_feedback, "JSON_Error")

    actionType:str = caller_data.get("tool")
    
    actionMethod = get_tool(tool_type_str=actionType)
    return (actionMethod(caller_data.get("input")),caller_data.get("tool"))




def summarizer(planner_question, currentplan):
    print("summarizes and shows user the output")


def get_action(action_type_str):
    for key, action in ActionTypes.items():
        if key.startswith(action_type_str):
            return action
            
    return lambda: "Action not recognized."
def get_tool(tool_type_str):
    for key, action in TOOLS.items():
        if key.startswith(tool_type_str):
            return action
            
    return lambda: "Tool not recognized."



def workflow(self,user_question, max_steps = 10):
    model = ChatOllama(model="llama3.1:8b", temperature=0)
    trajectory: List[Dict[str, Any]] = []
    print(PlannerPrompt)
    planner(self,model=model, planner_question=user_question)
    # plannerResult = model.invoke(f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{PlannerPrompt}<|eot_id|>
    # <|start_header_id|>user<|end_header_id|>{user_question}<|eot_id|>
    # <|start_header_id|>assistant<|end_header_id|>""")
    # plan:json = json.loads(plannerResult.content.strip())
    print(self.Plan)
    for step in self.Plan:
        actionType:str = step.get("ActionType")
        actionMethod = get_action(action_type_str=actionType)
        # if(actionMethod is str):
        #     print(actionMethod)
        # else:
        stateprompt = build_state_prompt(agent_query=step.get("context"),trajectory=trajectory,step=step.get("step"),max_steps=max_steps)

        (toolResult,usedTool) = actionMethod(model, stateprompt, step.get("toolType"))

        if usedTool == "Finish":
            print("Final Answer:", toolResult)
            break  # Exit the loop
            
        # Otherwise, append observation and continue

        trajectory.append({
                    "step": step.get("step"),
                    "action": step.get("ActionType"),
                    "tool_name": usedTool,
                    "tool_input": stateprompt,
                    "observation": toolResult,
                })
        print(toolResult) #Jag måste lägga in resultaten från tools sen!!

        
    
       

       
       

       


    # for step in range(1,max_steps + 1):
    #     userQuery = build_state_prompt(
    #         user_query=user_question,
    #         trajectory=trajectory,
    #         step=step,
    #         max_steps=max_steps,
    # )
#         plannerResult = model.invoke(f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{PlannerPrompt}<|eot_id|>
# <|start_header_id|>user<|end_header_id|>{user_question}<|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>""")
#         print(plannerResult.content)

# def ask_the_sir(user_question):
#     print(f"\n--- [INCOMING QUESTION]: {user_question} ---")

#     # STEG 1: RAG - Hämta metadata 
#     # Här söker vi i ChromaDB efter vilka kolumner som är relevanta för frågan.
#     relevant_info = get_schema_advice(user_question)


#     # STEG 2: PLANNER - Skapa SQL (Integration)
#     planner_prompt = f"""You are a Supply Chain Optimization Expert.
#     Table: 'sales_data'
#     Metadata: {relevant_info}

#     GOAL: Calculate the OPTIMAL inventory distribution for a specific category based on sales volume.

#     STRICT RULES:
#     1. Identify the 'product_category' requested.
#     2. Calculate the total units sold for that entire category.
#     3. Calculate the percentage (%) for each 'product_type' within that category.
#     4. Use this SQL structure:
#     SELECT 
#         product_type, 
#         SUM(transaction_qty) as units_sold,
#         ROUND(SUM(transaction_qty) * 100.0 / (SELECT SUM(transaction_qty) FROM sales_data WHERE product_category = 'KATEGORI'), 2) as optimal_distribution_pct
#     FROM sales_data
#     WHERE product_category = 'KATEGORI'
#     GROUP BY product_type
#     ORDER BY optimal_distribution_pct DESC;

#     5. Return ONLY the raw SQL query."""
    
#     sql_response = model.invoke([("system", planner_prompt), ("human", user_question)])
#     # RENSNING: Säkerställ att vi bara får SQL-koden
#     sql_query = sql_response.content.strip().replace("```sql", "").replace("```", "").strip()
#     # En extra säkerhetsåtgärd: ta bara det som börjar med SELECT
#     if "SELECT" in sql_query.upper():
#         sql_query = sql_query[sql_query.upper().find("SELECT"):]
#     print(f"-> [SIR'S PLAN]: Executing SQL: {sql_query}")

#     # STEG 3: EXECUTION - Hämta data 
#     # Vi kör SQL-frågan mot ecommerce_sales.db
#     raw_data = run_sql_query(sql_query)
#     print(f"-> [DATABASE RESULT]: {raw_data}")

#     # STEG 4: SUMMARIZER - Presentera resultatet 
#     summarizer_prompt = f"""You are a Senior Inventory Strategist. 
#     The user wants to know the OPTIMAL distribution for their stock.
#     Data from database: {raw_data}

#     TASK: 
#     1. Present the percentages as the "Recommended Stock Mix".
#     2. Explain that if they were to stock 100 units, they should buy exactly [X] of [Type A], [Y] of [Type B], etc.
#     3. Speak in an incredibly posh, authoritative, and helpful manner."""
    
#     final_response = model.invoke(summarizer_prompt)
    
#     print("\n--- [Final Response] ---")
#     print(final_response.content)

if __name__ == "__main__":
    # Denna fråga tvingar nu agenten att agera strategisk rådgivare 
    # och räkna ut fördelningen (distributionen) i procent.
    test_query = "My good Sir, based on our historical sales, what is the optimal percentage distribution of product types for the Coffee category in each of our stores?"
    workflow(Any, test_query)
    # ask_the_sir(test_query)