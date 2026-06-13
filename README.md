# 🧬 DataGenome AI — RAG Clinical Intelligence Chatbot

A fully free, API-key-free RAG chatbot for CDISC clinical trial data.

## Features
- 📋 Schema Explorer — all SDTM/ADaM table structures
- 🛡️ DQ Audit — data quality issues & remediations
- 📊 DQ Dashboard — completeness metrics & trends
- 🔗 ER Relationships — table join diagrams
- 📖 Data Dictionary — all 40+ column definitions
- 📚 GCP Glossary — 13 CDISC/regulatory terms
- 📜 Regulatory Report — FDA/EMA dossier summary
- ⚠️ Adverse Events — AE domain summary

## Tech Stack
- **Retrieval**: BM25 Okapi (rank-bm25) — zero cost, no GPU
- **Knowledge base**: 76 clinical chunks across 9 categories
- **UI**: Streamlit
- **LLM**: None required — deterministic RAG answer synthesis

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud (Free)
1. Fork/push this repo to GitHub
2. Go to share.streamlit.io → New app
3. Select your repo → `app.py`
4. Click Deploy — no secrets needed!
