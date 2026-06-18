# 🧬 DataGenome AI — Decoding the DNA of Life Sciences Data

**All 8 abstract capabilities delivered through a single conversational chatbot.**
No SQL · No API key · No cost · Streamlit Cloud free tier ready.

## Abstract Capabilities → Chat Commands

| Abstract Capability | Example Chat Commands |
|---|---|
| Schema Intelligence | "explain the patients table", "describe lab_results columns" |
| Business Glossary | "what is ALT?", "define SDTM", "what does GCP mean?" |
| DQ Audit | "show data quality issues", "missing values in patients" |
| ER Diagram | "show ER diagram", "draw entity relationship map" |
| Analytics Dashboard | "create a dashboard", "chart age distribution", "plot treatment arm" |
| Data Dictionary | "show data dictionary", "export data dictionary" |
| Regulatory PDF | "generate PDF report", "create regulatory report" |
| Data Q&A | "max age in patients", "mean ALT value", "distribution of severity" |

## Supported Data Sources
- CSV / TSV / TXT / Excel (.xlsx/.xls) / JSON / SQLite
- PostgreSQL / MySQL (via connection string)
- Multiple files simultaneously for cross-table analysis

## Quick Start
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy Free on Streamlit Cloud
1. Push to GitHub
2. share.streamlit.io → New app → select repo → app.py
3. Deploy — no secrets needed!

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI — upload panel, capability bar, chat |
| `core.py` | Data loading, profiling, ER SVG, Plotly dashboard, PDF, BM25 RAG |
| `responder.py` | NL intent router + all 8 HTML answer builders |
| `patients.csv` | Sample clinical trial patient data (30 subjects, 18 variables) |
| `lab_results.csv` | Sample laboratory results (50 records, 12 parameters) |
