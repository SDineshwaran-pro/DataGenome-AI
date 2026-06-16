# 🧬 DataGenome AI — Universal RAG Chatbot

Upload any dataset, connect a database, click **Analyse**, then chat about your data.
**No API key · No cost · Streamlit Cloud free tier compatible.**

## How to use
1. **Upload files** tab → drop CSV / Excel / JSON / SQLite → click 🔍 Analyse
2. **Connect Database** tab → enter SQLite path or PG/MySQL URI → click 🔌 Connect & Analyse
3. **CDISC Demo** tab → use built-in clinical trial demo instantly
4. Ask anything in the chat box below

## Supported formats
| Source | Format |
|---|---|
| Files | CSV, TSV, TXT, Excel (.xlsx / .xls), JSON, SQLite (.db / .sqlite) |
| Databases | SQLite file path, PostgreSQL URI, MySQL URI |
| Multiple | Upload many files at once — all analysed together |

## What the chatbot can answer
- *"What is the maximum salary?"*
- *"Show schema of the orders table"*
- *"Are there relationships between customers and orders?"*
- *"Show missing values in transactions"*
- *"Distribution of product_category"*
- *"Show all data quality issues"*
- *"What is the mean budget per department?"*
- *"Full statistics for the employees table"*

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy free on Streamlit Cloud
1. Push this folder to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select repo → branch → `app.py` → Deploy
4. No secrets or environment variables needed!

## Files
| File | Purpose |
|---|---|
| `app.py` | Main Streamlit UI — upload panel, chat, chips |
| `data_loader.py` | Universal file reader (CSV/Excel/JSON/SQLite/DB) |
| `analyzer.py` | Cross-dataset relationship & DQ detection engine |
| `dynamic_rag.py` | BM25 index builder from uploaded data |
| `dynamic_answer.py` | Answer generator for user datasets |
| `rag_engine.py` | Static BM25 index for CDISC demo |
| `answer_engine.py` | Answer generator for CDISC demo |
| `stats_engine.py` | Natural language statistics for CDISC demo |
| `data.py` | CDISC demo clinical data |
