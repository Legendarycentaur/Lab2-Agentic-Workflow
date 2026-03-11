# Coffee Sales Agentic Workflow

Detta projekt är en del av Laboration 2 och demonstrerar ett **Agentiskt Arbetsflöde** (Multi-Agent System). Systemet använder en arkitektur med **Planner**, **Caller** och **Summarizer** för att analysera försäljningsdata från en kaffebutik.

Systemet kombinerar **RAG** (Retrieval Augmented Generation) för att förstå databasschemat och en deterministisk **Calculator** för att säkerställa 100% korrekta matematiska beräkningar och undvika LLM-hallucinationer.

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

1.  **Skapa och aktivera miljön:**
    ```bash
    conda create -n coffee_agent python=3.12 -y
    conda activate coffee_agent
    ```
2.  **Installera beroenden via requirements.txt:**
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

Se till att Ollama körs i bakgrunden.

1.  **Starta huvudagenten:**
    ```bash
    python agentic_main.py
    ```

### Systemarkitektur
Systemet bygger på en iterativ loop där varje steg loggas för full spårbarhet:



* **Planner:** Ansvarar för den strategiska analysen och väljer nästa steg baserat på historiken.
* **Caller:** En teknisk översättare som mappar planer till specifika verktygsanrop och rensar bort syntaktiskt brus.
* * **Summarizer:** När målet är uppnått sammanställer denna roll all insamlad data till en formell, mänskligt läsbar rapport.
* **Memory Management:** Använder en "Sliding Window"-metod för att undvika att agenten blir överväldigad av gammal historik och fastnar i loopar.

---

## Projektets filer och stack

### Teknisk Stack
* **LLM:** Ollama (Llama 3.1 8B)
* **Framework:** LangChain (Core, Community, Chroma, Ollama)
* **Database:** SQLite & SQLAlchemy
* **Vector DB:** ChromaDB

### Filstruktur
* `agentic_main.py`: Huvudfilen med Multi-Agent-logiken och Controller-loopen.
* `data_ingestion.py`: Skript för automatiserad datahämtning och SQL-initiering.
* `vector_store.py`: Hanterar embeddings för databasschemat (RAG).
* `requirements.txt`: Lista på alla nödvändiga Python-bibliotek.

---

## Felsökning

* **"Failed to determine an answer to the question.":** Detta händer om agenten når max antal försök (retry_count).
* **Loopar:** Om agenten repeterar samma steg, prova att rensa gamla loggfiler och starta om Ollama.
* **Requirements:** Om `pysqlite3-binary` orsakar problem på Windows, se till att du har "Build Tools for Visual Studio" installerade.
