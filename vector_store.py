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
    "The table name is 'sales_data'. Use this for all e-commerce queries.",
    "Column 'order_id' is a BIGINT and serves as the unique identifier for each sale.",
    "Column 'order_date' is TEXT in YYYY-MM-DD format. Use this for filtering by time or date.",
    "Column 'sku' is TEXT and stands for Stock Keeping Unit, identifying unique products.",
    "Column 'color' is TEXT and describes the product's color variant.",
    "Column 'size' is TEXT and describes the product's size (e.g., S, M, L, XL).",
    "Column 'unit_price' is a BIGINT representing the price of a single item.",
    "Column 'quantity' is a BIGINT representing how many items were bought in one order.",
    "Column 'revenue' is a BIGINT representing the total money from the order (unit_price * quantity)."
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
    docs = db.similarity_search(user_query, k=2)
    return "\n".join([d.page_content for d in docs])

if __name__ == "__main__":
    # Detta körs bara när du startar just denna fil
    initialize_metadata()
    
    # Ett snabbt test för att se att det funkar
    print("\n--- Testar RAG-sökning ---")
    print(f"Fråga: 'How much money?' -> Svar: {get_schema_advice('How much money?')}")