# Coffee Sales Agentic Workflow

Detta projekt är en del av Laboration 2 och demonstrerar ett **Agentiskt Arbetsflöde** (Multi-Agent System). Systemet använder en arkitektur med **Planner**, **Caller** och **Summarizer** för att analysera försäljningsdata från en kaffebutik.

Systemet kombinerar **RAG** (Retrieval Augmented Generation) för att förstå databasschemat, SQLite databas som komminuceras med via en **sql-tool** och en deterministisk **Calculator** för att säkerställa korrekta matematiska beräkningar och undvika LLM-hallucinationer.

---

## 1. Förberedelser: Ollama

Innan du startar agenten måste Ollama-motorn vara installerad och rätt modell finnas tillgänglig lokalt.

1.  **Installera Ollama:** Ladda ner och installera från [ollama.com](https://ollama.com/).
2.  **Ladda ner modellen:** Kör följande kommando i din terminal:
    ```bash
    ollama pull llama3.1:8b
    ```
    *Modellen måste vara nedladdad för att Python-skripten ska kunna kommunicera med LLM:en.*

---

## 2. Sätt upp Conda-miljön

För att säkerställa kompatibilitet används Python 3.12.
1.  **Installera Conda:** Ladda ner och installera från [conda.io](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2.  **Skapa och aktivera miljön:**
    ```bash
    conda create -n coffee_agent python=3.12 -y
    conda activate coffee_agent
    ```
3.  **Installera beroenden via requirements.txt:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 3. Dataprocessering (Ingestion)

Projektet använder automatiserad datahämtning och transformation för att bygga den lokala databasen.

1.  **Kör dataingestion:**
    ```bash
    python data_ingestion.py
    ```
    *Detta skript använder `kagglehub` för att hämta datasetet och `sqlalchemy` för att skapa SQL-databasen `coffee_sales.db` med tabellen `sales_data`.*

---

## 4. Vector Store & RAG (Embeddings)

För att agenten ska kunna navigera i databasen används en Vector Store (`langchain-chroma`) som lagrar semantisk information om tabellens schema.

1.  **Indexera schemat:**
    ```bash
    python vector_store.py
    ```
    *Detta gör att agenten kan använda verktyget `get_schema` för att hitta korrekta kolumnnamn som `product_category` och `product_type`.*

---

## 5. Kör Arbetsflödet

1.  **Starta huvudagenten:**
    ```bash
    python agentic_main.py
    ```

### Systemarkitektur
Systemet bygger på en iterativ loop där varje steg loggas för full spårbarhet:



* **Planner:** Ansvarar för den strategiska analysen och väljer nästa steg baserat på historiken.
* **Caller:** En teknisk översättare som mappar planen till specifikt verktygsanrop.
* **Summarizer:** När målet är uppnått sammanställer denna roll all insamlad data till en formell, mänskligt läsbar rapport.

---

### Filstruktur
* `agentic_main.py`: Huvudfilen med Multi-Agent-logiken och Controller-loopen.
* `data_ingestion.py`: Skript för automatiserad datahämtning och SQL-initiering.
* `vector_store.py`: Hanterar embeddings för databasschemat (RAG).
* `requirements.txt`: Lista på alla nödvändiga Python-bibliotek.
