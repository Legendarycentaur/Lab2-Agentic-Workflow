from langchain_ollama import ChatOllama

def main():
    model = ChatOllama(model="llama3.1:8b", temperature=100)
    response = model.invoke("Hello my good sir")
    print(response.content)


    


if __name__ == "__main__":
    main()