# Coffee Sales Agentic Workflow

Detta projekt är en del av Laboration 2 och demonstrerar ett **Agentiskt Arbetsflöde** (Multi-Agent System). Systemet använder en arkitektur med **Planner**, **Caller** och **Summarizer** för att analysera försäljningsdata från en kaffebutik.

Systemet kombinerar **RAG** (Retrieval Augmented Generation) för att förstå databasschemat och en deterministisk **Calculator** för att säkerställa korrekta matematiska beräkningar (och undvika hallucinationer).

---

## 1. Sätt upp Conda-miljön

För att säkerställa att alla bibliotek fungerar korrekt rekommenderas en isolerad miljö.

1.  **Skapa miljön:**
    ```bash
    conda create -n coffee_agent python=3.12 -y
    ```
2.  **Aktivera miljön:**
    ```bash
    conda activate coffee_agent
    ```
3.  **Installera beroenden:**
    ```bash
    pip install langchain langchain-ollama pysqlite3 sentence-transformers chromadb pandas
    ```

---

## 📥 2. Dataset och Dataprocessering (Ingestion)

Projektet bygger på datasetet *Coffee Shop Sales*.

1.  **CSV-data:** Placera din rådatafil i rotkatalogen.
2.  **Skapa SQL-databas:** Kör skriptet för att läsa in CSV-data till SQLite.
    ```bash
    python data_ingestion.py
    ```
    *Detta skapar filen `coffee_sales.db` med tabellen `sales_data` genom att tvätta och transformera rådatan.*

---

## 🔍 3. Vector Store & RAG (Embeddings)

För att agenten ska veta vilka kolumner som finns (t.ex. `product_category` istället för bara `category`) används en Vector Store för att lagra metadata om schemat.

1.  **Indexera schemat:** Kör skriptet som skapar embeddings för databasschemat.
    ```bash
    python vector_store.py
    ```
    *Detta gör att agenten kan använda verktyget `get_schema` för att hämta teknisk information via semantisk sökning i ChromaDB.*

---

## 🤖 4. Kör Arbetsflödet

Se till att **Ollama** är igång lokalt och att modellen `llama3.1:8b` är tillgänglig.

1.  **Starta agenten:**
    ```bash
    python agentic_main.py
    ```

### Systemets logik:

* **Planner:** Bryter ner målet till tekniska steg och hanterar fel (t.ex. om en SQL-fråga misslyckas).
* **Caller:** Översätter instruktioner till rena verktygsanrop (`run_sql`, `calculator`, `get_schema`).
* **Sliding Window Memory:** Agenten ser endast de senaste händelserna för att undvika "loop-hypnos" och fokusera på den mest relevanta observationen.

---

## 📂 Filbeskrivning

| Fil | Beskrivning |
| :--- | :--- |
| `agentic_main.py` | Huvudfilen med Multi-Agent-logiken och Controller-loopen. |
| `data_ingestion.py` | Skript för att läsa in CSV-data och skapa SQL-databasen. |
| `vector_store.py` | Hanterar embeddings och sökning i databasschemat (RAG). |
| `sql_tool.py` | Innehåller funktionen för att exekvera SQL-frågor mot SQLite. |

---

## ⚠️ Felsökning (Troubleshooting)

* **"Failed to determine an answer to the question.":** Detta händer om agenten når max antal försök. 
* **JSON Errors:** Systemet har inbyggda "fail-safes" som fångar upp om LLM:en genererar ogiltig JSON.
* **Matematiska hallucinationer:** Om agenten räknar fel (t.ex. summan blir 135%), tvingar systemet den att använda `calculator`-verktyget i nästa steg.
