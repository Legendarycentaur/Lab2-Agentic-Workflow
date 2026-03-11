from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os


DB_PATH = "./chroma_langchain_db" # Local directory where the vector database is saved persistently
EMBED_MODEL = "nomic-embed-text" # The Ollama embed-model used to convert text into vector embeddings

def get_vector_db():
    """Loads the existing vector database from the local disk.
    If the directory does not exist, Chroma will prepare to create a new one."""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

def initialize_metadata():
    """
    Populates the vector database with knowledge about the SQL tables.
    This acts as the foundation for the Retrieval-Augmented Generation (RAG) system.
    """
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    
    # Define the schema descriptions here. 
    # This is the 'brain' of the agent, providing context about available data.
    descriptions = [
    "The table name is 'sales_data'.",
    "Column 'product_category': string, is the main category (e.g., 'Coffee', 'Tea').",
    "Column 'product_type': string, represents the specific variants of products within a category.",
    "Column 'store_id': int and 'store_location': string, identify the specific shop.",
    "Column 'transaction_qty': int, is the number of units sold.",
    "To calculate store-specific distribution: Group by 'product_type' and 'store_location'.",
    "Formula: (SUM of units for a specific type in a store / SUM of total units for that category in that same store) * 100."
]
    
    # Create the vector database from the text list and save it to the specified directory
    vector_db = Chroma.from_texts(
        texts=descriptions,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print("✓ Metadata vectorized and stored in chroma_langchain_db.")

def get_schema_advice(user_query):
    """Retrieves the most relevant schema descriptions based on a specific query.
    This function is exposed as a 'Tool' to the AI Agent."""
    db = get_vector_db()
    # Perform a similarity search to fetch the top 5 (k=5) most relevant documents for the query
    docs = db.similarity_search(user_query, k=5)
    # Combine the retrieved documents into a single readable string for the agent
    return "\n".join([d.page_content for d in docs])

if __name__ == "__main__":
    initialize_metadata()
    