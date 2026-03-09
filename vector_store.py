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
    "The table name is 'sales_data'. It contains coffee shop transaction records.",
    "Column 'transaction_id' is a unique BIGINT identifier for each sale.",
    "Column 'transaction_date' is TEXT (YYYY-MM-DD). Use this for daily or monthly sales analysis.",
    "Column 'transaction_time' is TEXT (HH:MM:SS) for time-of-day analysis.",
    "Column 'transaction_qty' (BIGINT) is the number of items sold in one transaction.",
    "Column 'store_id' and 'store_location' (TEXT) identify where the sale happened (e.g., Manhattan, Astoria).",
    "Column 'product_id' is a unique BIGINT for each item.",
    "Column 'unit_price' is a FLOAT representing the cost of a single item.",
    "Column 'product_category' (TEXT) is the broad category like 'Coffee', 'Tea', or 'Bakery'.",
    "Column 'product_type' (TEXT) is the specific style, e.g., 'Gourmet brewed coffee'.",
    "Column 'product_detail' (TEXT) is the exact product name, e.g., 'Ethiopia Roasting'.",
    "To calculate total revenue for a row, use: (transaction_qty * unit_price)."
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