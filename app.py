"""
DataGenome AI — Full RAG Chatbot (No API Key Required)
BM25 retrieval + intelligent answer synthesis
Run: streamlit run app.py
"""
import streamlit as st
from rag_engine import ClinicalRAG
from answer_engine import generate_answer

st.set_page_config(
    page_title="DataGenome AI",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────── CSS ─────────────────────────────────────────────
st.markdown("""
<style>
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 0 !important; max-width: 100% !important; }

  /* Top bar */
  .topbar {
    background: linear-gradient(135deg, #0f172a 0%, #134e4a 100%);
    color: white; padding: 0.55rem 1.25rem;
    display: flex; align-items: center; gap: 0.65rem;
  }
  .topbar-title { font-size: 1.05rem; font-weight: 800; letter-spacing: -0.3px; }
  .topbar-sub   { font-size: 0.7rem; opacity: 0.45; margin-left: 2px; }
  .topbar-right { margin-left: auto; font-size: 0.7rem; opacity: 0.55;
                  display: flex; align-items: center; gap: 5px; }
  .dot { width: 7px; height: 7px; border-radius: 50%;
         background: #22c55e; display: inline-block; }

  /* Chat bubbles */
  .bubble-user {
    background: #0d9488; color: white;
    padding: 0.58rem 0.95rem; border-radius: 14px 14px 3px 14px;
    margin: 0.4rem 0 0.4rem auto; max-width: 78%; width: fit-content;
    font-size: 0.87rem; line-height: 1.5; display: block;
  }
  .bubble-ai {
    background: #ffffff; color: #1e293b;
    border: 1px solid #e2e8f0; border-left: 3px solid #0d9488;
    padding: 0.75rem 1rem; border-radius: 3px 14px 14px 14px;
    margin: 0.4rem 0; max-width: 96%; font-size: 0.86rem;
    line-height: 1.65; display: block;
  }
  .bubble-ai strong { color: #0f172a; }
  .bubble-ai code {
    background: #f1f5f9; color: #0f766e;
    padding: 1px 5px; border-radius: 4px; font-size: 0.78rem;
  }
  .bubble-ai pre {
    background: #f1f5f9; padding: 6px 10px;
    border-radius: 6px; font-size: 0.74rem;
    overflow-x: auto; margin-top: 5px;
  }

  /* Tags */
  .tag { display:inline-block; padding:1px 7px; border-radius:20px;
         font-size:0.68rem; font-weight:700; margin:1px; }
  .ts  { background:#ccfbf1; color:#0f766e; }
  .ta  { background:#dbeafe; color:#1d4ed8; }
  .tc  { background:#fee2e2; color:#dc2626; }
  .tw  { background:#fef9c3; color:#ca8a04; }
  .ti  { background:#dbeafe; color:#2563eb; }
  .tok { background:#dcfce7; color:#15803d; }

  /* Stat cards */
  .sc-row { display:flex; gap:5px; flex-wrap:wrap; margin:6px 0; }
  .sc { background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px;
        padding:5px 10px; min-width:76px; }
  .sc-l { font-size:0.64rem; color:#64748b; margin-bottom:1px; }
  .sc-v { font-size:0.95rem; font-weight:700; color:#0f172a; }

  /* Mini bar */
  .bar-row { display:flex; align-items:center; gap:6px; margin:3px 0; }
  .bar-bg  { flex:1; background:#f1f5f9; border-radius:4px;
             height:11px; overflow:hidden; }
  .bar-fill { height:100%; border-radius:4px; }

  /* Sources strip */
  .sources {
    background: #f8fafc; border: 1px solid #e9eef4;
    border-radius: 0 0 8px 8px; padding: 5px 10px;
    font-size: 0.7rem; color: #94a3b8;
    display: flex; flex-wrap: wrap; gap: 4px; align-items: center;
  }
  .src-chip {
    background: #e0f2fe; color: #0369a1;
    border-radius: 10px; padding: 1px 7px;
    font-size: 0.67rem; font-weight: 600;
  }
  .src-chip.dq         { background:#fee2e2; color:#dc2626; }
  .src-chip.schema     { background:#ccfbf1; color:#0f766e; }
  .src-chip.column     { background:#ccfbf1; color:#0f766e; }
  .src-chip.glossary   { background:#fef9c3; color:#854d0e; }
  .src-chip.er         { background:#dbeafe; color:#1d4ed8; }
  .src-chip.regulatory { background:#f3e8ff; color:#7c3aed; }
  .src-chip.ae         { background:#fee2e2; color:#c2410c; }
  .src-chip.analysis   { background:#dcfce7; color:#15803d; }
  .src-chip.study      { background:#f0f9ff; color:#0369a1; }

  /* Action buttons */
  div[data-testid="stButton"] > button {
    width: 100%; text-align: left !important;
    justify-content: flex-start !important;
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    color: #334155 !important;
    font-size: 0.8rem !important;
    padding: 0.4rem 0.75rem !important;
    transition: all 0.12s;
  }
  div[data-testid="stButton"] > button:hover {
    border-color: #0d9488 !important;
    background: #ffffff !important;
    color: #0f172a !important;
  }

  /* Send button */
  .send-wrap div[data-testid="stButton"] > button {
    background: #0d9488 !important;
    color: white !important; border: none !important;
    font-weight: 700 !important; border-radius: 10px !important;
    justify-content: center !important; font-size: 0.88rem !important;
  }
  .send-wrap div[data-testid="stButton"] > button:hover {
    background: #0f766e !important;
  }

  /* Quick chips */
  .chip-wrap div[data-testid="stButton"] > button {
    border-radius: 20px !important;
    font-size: 0.71rem !important;
    padding: 0.18rem 0.7rem !important;
    background: #f1f5f9 !important;
    color: #475569 !important;
    width: auto !important;
  }
  .chip-wrap div[data-testid="stButton"] > button:hover {
    background: #e0f2fe !important;
    border-color: #7dd3fc !important;
  }

  /* Text input */
  .stTextArea textarea {
    border-radius: 10px !important;
    border: 1.5px solid #e2e8f0 !important;
    font-size: 0.87rem !important;
    resize: none !important;
    background: #fafafa !important;
  }
  .stTextArea textarea:focus {
    border-color: #0d9488 !important;
    background: #fff !important;
  }

  /* RAG badge */
  .rag-bar {
    display: flex; flex-wrap: wrap; gap: 4px;
    padding: 5px 1rem; background: #f8fafc;
    border-bottom: 1px solid #e9eef4;
    align-items: center;
  }
  .rag-label { font-size:0.67rem; color:#94a3b8; font-weight:700;
               text-transform:uppercase; letter-spacing:.05em; margin-right:2px; }

  /* Welcome card */
  .welcome {
    text-align: center; padding: 1.5rem 1rem;
    color: #64748b;
  }
  .welcome-title { font-size:1rem; font-weight:800; color:#0f172a; margin-bottom:5px; }
  .welcome-sub   { font-size:0.82rem; line-height:1.6; max-width:520px; margin:0 auto 1rem; }
  .tip-chip {
    display:inline-block; background:#f1f5f9; border-radius:8px;
    padding:4px 10px; font-size:0.75rem; margin:3px;
    color:#475569; border:1px solid #e2e8f0;
  }

  /* Chat scroll area */
  .chat-area { padding: 0.5rem 1.25rem 0.3rem; }

  /* Scrollbar styling */
  ::-webkit-scrollbar { width: 5px; height: 5px; }
  ::-webkit-scrollbar-track { background: #f8fafc; }
  ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────── Init ────────────────────────────────────────────
@st.cache_resource(show_spinner="🧬 Building RAG knowledge index…")
def load_rag():
    return ClinicalRAG()

rag   = load_rag()
stats = rag.stats()

CHIPS = [
    ("🛡️ DQ Audit",       "Show all data quality issues and remediations"),
    ("📋 SDTM Tables",    "List all SDTM and ADaM tables"),
    ("🔗 ER Diagram",     "Explain ER relationships between all tables"),
    ("📖 Dictionary",     "Show full data dictionary for all columns"),
    ("📚 Glossary",       "List all CDISC and GCP glossary terms"),
    ("📊 Dashboard",      "Show DQ completeness metrics and error trends"),
    ("📜 Reg Report",     "Show regulatory submission status and dossier"),
    ("⚠️ AE Summary",    "Summarise AE domain data and severity breakdown"),
]

# Session state
if "messages"  not in st.session_state: st.session_state.messages  = []
if "pending"   not in st.session_state: st.session_state.pending   = ""
if "input_key" not in st.session_state: st.session_state.input_key = 0


# ─────────────────────────── Submit logic ────────────────────────────────────
def submit(q: str):
    q = q.strip()
    if not q:
        return
    # Retrieve
    chunks = rag.retrieve(q, top_k=6)
    # Generate answer
    html, btns = generate_answer(q, chunks)
    # Store
    st.session_state.messages.append({"role": "user",  "text": q})
    st.session_state.messages.append({
        "role": "ai", "html": html, "btns": btns,
        "sources": chunks,
    })
    st.session_state.pending   = ""
    st.session_state.input_key += 1   # clears textarea


# Handle pending (chip/button clicks submitted before render)
if st.session_state.pending:
    submit(st.session_state.pending)


# ─────────────────────────── Top bar ─────────────────────────────────────────
st.markdown(f"""
<div class="topbar">
  <span style="font-size:1.25rem">🧬</span>
  <span class="topbar-title">DataGenome AI</span>
  <span class="topbar-sub">RAG · CDISC Clinical Intelligence · No API Key Required</span>
  <div class="topbar-right">
    <span class="dot"></span>
    STUDY-2026-FIBER &nbsp;·&nbsp; {stats['total']} chunks
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────── RAG index bar ───────────────────────────────────
chip_html = "".join(
    f'<span class="src-chip {cat}">{cat} · {n}</span>'
    for cat, n in sorted(stats["by_category"].items())
)
st.markdown(
    f'<div class="rag-bar"><span class="rag-label">RAG Index</span>'
    f'{chip_html}'
    f'<span style="margin-left:auto;font-size:0.67rem;color:#cbd5e1">'
    f'BM25 Okapi · Free · Local · No API key</span></div>',
    unsafe_allow_html=True
)

# ─────────────────────────── Quick chips ─────────────────────────────────────
chip_cols = st.columns(len(CHIPS))
for col, (label, query) in zip(chip_cols, CHIPS):
    with col:
        st.markdown('<div class="chip-wrap">', unsafe_allow_html=True)
        if st.button(label, key=f"chip_{label}"):
            st.session_state.pending = query
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr style='margin:0;border-color:#e9eef4'>", unsafe_allow_html=True)

# ─────────────────────────── Welcome screen ──────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome">
      <div style="font-size:2.2rem;margin-bottom:0.4rem">🧬</div>
      <div class="welcome-title">DataGenome AI — RAG Clinical Intelligence</div>
      <div class="welcome-sub">
        Ask anything about <strong>STUDY-2026-FIBER</strong> in plain English.<br>
        BM25 retrieval finds the right knowledge chunks — no API key, no internet, no cost.
      </div>
      <div>
        <span class="tip-chip">💡 "What DQ issues are blocking submission?"</span>
        <span class="tip-chip">💡 "What columns does the LB table have?"</span>
        <span class="tip-chip">💡 "What is ALCOA+ and why does it matter?"</span>
        <span class="tip-chip">💡 "Explain the difference between ITT and SAF"</span>
        <span class="tip-chip">💡 "What is the ER relationship between DM and AE?"</span>
        <span class="tip-chip">💡 "Show me all critical DQ issues in the DM table"</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────── Chat messages ───────────────────────────────────
st.markdown('<div class="chat-area">', unsafe_allow_html=True)

for idx, msg in enumerate(st.session_state.messages):

    if msg["role"] == "user":
        st.markdown(
            f'<div class="bubble-user">👤 {msg["text"]}</div>',
            unsafe_allow_html=True
        )

    else:
        # AI bubble
        st.markdown(
            f'<div class="bubble-ai">🧬 {msg["html"]}</div>',
            unsafe_allow_html=True
        )

        # Sources strip
        sources = msg.get("sources", [])
        if sources:
            chips = "".join(
                f'<span class="src-chip {s["category"]}" title="Score: {s["score"]}">'
                f'{s["title"][:42]}{"…" if len(s["title"])>42 else ""}'
                f'</span>'
                for s in sources[:5]
            )
            st.markdown(
                f'<div class="sources">📎 Retrieved: {chips}</div>',
                unsafe_allow_html=True
            )

        # Action buttons
        btns = msg.get("btns", [])
        if btns:
            ncols = min(len(btns), 4)
            btn_cols = st.columns(ncols)
            for j, (lbl, q) in enumerate(btns):
                with btn_cols[j % ncols]:
                    if st.button(lbl, key=f"btn_{idx}_{j}"):
                        st.session_state.pending = q
                        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<hr style='margin:0;border-color:#e9eef4'>", unsafe_allow_html=True)

# ─────────────────────────── Input row ───────────────────────────────────────
in_col, btn_col = st.columns([5, 1])
with in_col:
    user_input = st.text_area(
        "msg",
        label_visibility="collapsed",
        placeholder="Ask about schemas, DQ issues, ER relationships, glossary, regulatory…  (Enter to send)",
        height=66,
        key=f"input_{st.session_state.input_key}",
    )
with btn_col:
    st.markdown("<div class='send-wrap'>", unsafe_allow_html=True)
    send = st.button("Send ▶", key="send_btn", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

if send and user_input and user_input.strip():
    submit(user_input)
    st.rerun()

# Enter key support via JS
st.markdown("""
<script>
const ta = window.parent.document.querySelector('textarea');
if (ta) {
  ta.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      const btn = window.parent.document.querySelector('button[kind="secondary"]');
      if (btn) btn.click();
    }
  });
}
</script>
""", unsafe_allow_html=True)

st.markdown(
    "<div style='text-align:center;font-size:0.67rem;color:#cbd5e1;"
    "padding:0.3rem 0 0.5rem'>"
    "BM25 Okapi retrieval · 76 clinical knowledge chunks · "
    "Zero cost · No API key · Runs on Streamlit Cloud free tier"
    "</div>",
    unsafe_allow_html=True,
)
