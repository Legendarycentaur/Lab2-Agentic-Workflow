from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os

# Inställningar för On-Prem (Spara RAM genom att skriva till disk)
DB_PATH = "./chroma_langchain_db"
EMBED_MODEL = "nomic-embed-text"

def get_vector_db():
    """Laddar existerande databas från disk eller skapar en ny."""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

def initialize_metadata():
    """Fyller databasen med kunskap om SQL-tabellerna."""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    
    # Här beskriver du din data. Detta är 'hjärnan' i din RAG.
    descriptions = [
    "The table name is 'sales_data'.",
    "Column 'product_category' is the main category (e.g., 'Coffee', 'Tea').",
    "Column 'product_type' represents the specific variants of products within a category.",
    "Column 'store_id' and 'store_location' identify the specific shop.",
    "Column 'transaction_qty' is the number of units sold.",
    "To calculate store-specific distribution: Group by 'product_type' and 'store_location'.",
    "Formula: (SUM of units for a specific type in a store / SUM of total units for that category in that same store) * 100."
]
    
    # Skapa vektordatabasen och spara den i mappen chroma_langchain_db
    vector_db = Chroma.from_texts(
        texts=descriptions,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print("✓ Metadata vectorized and stored in chroma_langchain_db.")

def get_schema_advice(user_query):
    """Hämtar de mest relevanta beskrivningarna för en specifik fråga."""
    db = get_vector_db()
    # Vi hämtar de 2 mest relevanta meningarna baserat på vad användaren frågar om
    docs = db.similarity_search(user_query, k=5)
    return "\n".join([d.page_content for d in docs])

if __name__ == "__main__":
    # Detta körs bara när du startar just denna fil
    initialize_metadata()
    