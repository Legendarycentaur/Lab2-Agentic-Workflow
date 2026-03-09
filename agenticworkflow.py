import json
import sqlite3
from typing import Any, Dict, List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import sql_database
from langchain_ollama import OllamaEmbeddings
from sqlalchemy import create_engine
import datetime

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
    
    
    # engine = create_engine("sqlite:///ecommerce_sales.db")
    # conn = engine.connect()
    # conn.execute()
    conn = sqlite3.connect("ecommerce_sales.db")
    cursor = conn.cursor()
    # sqldocuments = sql_database.SQLDatabaseLoader(db=cursor,query="SELECT * " \
    # "FROM sales_data").load_and_split()

    cursor.execute("SELECT * FROM sales_data")
    rows = cursor.fetchall()
    
    documents = []
    for row in rows:
        order_id,order_date,sku,color,size,unit_price,quantity, revenue = row
    
        print(sku)
        print(color)
        doc = Document(
            page_content=f"sku={sku}, color={color}, size={size}, unitprice={unit_price}, quanityOnOrder={quantity}, revenueonorder={revenue}",
            metadata={
                "source_id": order_id,
                "date": order_date, 
                "database": "ecommerence_sales"
            }   
        )
        documents.append(doc)
    
    print(f"Split blog post into {len(documents)} sqldocuments.")

    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
    vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )
    vector_store.add_documents(documents=documents)

    results = vector_store.similarity_search(
    "Which is the most popular color",
    k=3
    )
    print(results)

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