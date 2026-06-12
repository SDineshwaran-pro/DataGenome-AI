# 🧬 DataGenome AI — Streamlit Prototype

Conversational Clinical Data Intelligence platform converted to a **100% free** Streamlit app.
**No API keys. No cloud costs. Runs locally.**

## Features

| Module | Description |
|---|---|
| 💬 AI Chatbot | Rule-based clinical Q&A (no API required) |
| 📋 Schema Explorer | Browse SDTM/ADaM table structures & columns |
| 🔗 ER Diagram | Interactive entity-relationship visualization |
| 🛡️ DQ Audit | Data quality issues with remediation guidance |
| 📊 DQ Dashboard | Charts: completeness, trends, severity breakdown |
| 📖 Data Dictionary | Searchable & exportable column definitions |
| 📜 Regulatory Report | Study summary + DQ compliance dossier |
| 📚 GCP Glossary | MedDRA, SDTM, ADaM, GCP term definitions |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

## Deploy Free on Streamlit Cloud

1. Push this folder to a GitHub repository
2. Go to https://share.streamlit.io
3. Connect your GitHub repo → select `app.py`
4. Click **Deploy** — free hosting, no credit card needed

## Stack

- **Frontend**: Streamlit
- **Charts**: Plotly
- **Data**: Pandas (in-memory mock CDISC data)
- **AI**: Rule-based NLP (upgrade to Claude/Gemini by adding API key)

## Upgrade to Real AI (Optional)

To connect to Claude API, replace the rule-based response logic in `app.py`
around the `# Generate rule-based response` comment with:

```python
import anthropic
client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": user_input}]
)
answer = response.content[0].text
```

Store your API key in `.streamlit/secrets.toml`:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```
