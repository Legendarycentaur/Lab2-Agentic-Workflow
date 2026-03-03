from langchain_ollama import ChatOllama
import datetime

SYSTEM_PROMPT = f"You are a fine Sir. You Speak in a posh manner and you will always answer with the finest of language. I am also a sir And the time is {datetime.datetime.now()}"

def main():
    model = ChatOllama(model="llama3.1:8b", temperature=100)
    response = model.invoke([("system", SYSTEM_PROMPT), ("human", "Hello My good sir")])
    print(response.content)


if __name__ == "__main__":
    main()