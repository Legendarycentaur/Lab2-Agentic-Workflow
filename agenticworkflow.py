import json
from typing import Any, Dict, List

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
import datetime

# SYSTEM_PROMPT = f"You are a fine Sir. Joking is your motto, You Speak in a posh manner and you will always answer with the finest of language. I am also a sir And the time is {datetime.datetime.now()}"
SYSTEM_PROMPT_First_Step = """You are an agent policy.
Decide one next action from current state.
 final answer can contain a text string with an answer to what the user wants answered.
Return ONLY JSON with this schema:
{
  "action": "tool" | "final",
  "tool_name": "GetSalesData" | "CalculateMostSoldItems" | null,
  "tool_input": "<number>" | "<string>" | null,
  "final_answer": "<string>" | null
}

The tools you can shoose from is only these: 
 - GetSalesData and it only takes the a sql query as input it can only look like this: SELECT size, SUM(quantity) as total_sold FROM sales_data GROUP BY size ORDER BY total_sold DESC LIMIT 5. 
 


Date is:
""" + str(datetime.datetime.now())

def build_state_prompt(user_query: str, trajectory: List[Dict[str, Any]], step: int, max_steps: int) -> str:
    return (
        f"User query: {user_query}\n"
        f"Step: {step}/{max_steps}\n"
        f"Trajectory so far (JSON): {json.dumps(trajectory)}\n"
        "Choose the next action."
    )


def state_to_messages(state: dict) -> list[tuple[str, str]]:
    prompt = build_state_prompt(
        user_query=state["user_query"],
        trajectory=state["trajectory"],
        step=state["step"],
        max_steps=state["max_steps"],
    )
    return [("system", SYSTEM_PROMPT_First_Step), ("human", prompt)]

def main(user_query: str, max_steps:int = 10):
    model = ChatOllama(model="llama3.1:8b", temperature=0, verbose=True)
    parser = JsonOutputParser()
    trajectory: List[Dict[str, Any]] = []

    policy_chain = RunnableLambda(state_to_messages) | model | parser 

    state = {
        "user_query": """How manny items are we going to sell next month so in april?:
    """,
        "trajectory": trajectory,
        "step": 1,
        "max_steps": 6,
    }
    decision = policy_chain.invoke(state)
    print("Decision:", decision)
    # for i in range(1,max_steps+1):
    #     modelresponse = model.invoke([("system", SYSTEM_PROMPT), ("human", user_query)]);


    


if __name__ == "__main__":
    main("My good sir what you up to at this time of day ☕")