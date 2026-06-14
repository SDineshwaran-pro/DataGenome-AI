# 🧬 DataGenome AI — Universal RAG Chatbot

Upload any dataset, connect a database, click Analyse, then chat about your data.
No API key · No cost · Streamlit Cloud free tier compatible.

## Supported data sources
| Source | Format |
|--------|--------|
| Files | CSV, TSV, TXT, Excel (.xlsx/.xls), JSON, SQLite (.db) |
| Databases | SQLite (file path), PostgreSQL, MySQL |

## How it works
1. Upload files or connect DB in the sidebar
2. Click **🔍 Analyse Datasets**
3. Ask anything in plain English

## Example queries after upload
- "What is the maximum age in employees?"
- "Show schema of the orders table"
- "Are there any missing values?"
- "Show relationships between all tables"
- "Distribution of status column"
- "Show all DQ issues"

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy free on Streamlit Cloud
1. Push repo to GitHub
2. Go to share.streamlit.io → New app → select repo → app.py
3. Deploy — no secrets needed
