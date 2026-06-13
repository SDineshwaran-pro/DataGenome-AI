"""
DataGenome AI — Full RAG Chatbot
BM25 retrieval + RAG answer synthesis + Natural language statistics
No API key · Runs on Streamlit Cloud free tier
"""
import streamlit as st
from rag_engine import ClinicalRAG
from answer_engine import generate_answer
from stats_engine import answer_stats_query

st.set_page_config(
    page_title="DataGenome AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────── CSS ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Top bar ── */
.topbar {
    background: linear-gradient(135deg,#0f172a 0%,#134e4a 100%);
    color: white;
    padding: 0.55rem 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    flex-wrap: wrap;
}
.topbar-title { font-size: 1rem; font-weight: 800; }
.topbar-sub   { font-size: 0.68rem; opacity: 0.5; }
.topbar-right {
    margin-left: auto;
    font-size: 0.68rem;
    opacity: 0.6;
    display: flex;
    align-items: center;
    gap: 5px;
}
.dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #22c55e;
    display: inline-block;
}

/* ── Chat bubbles ── */
.bubble-user {
    background: #0d9488;
    color: #ffffff !important;
    padding: 0.6rem 1rem;
    border-radius: 14px 14px 3px 14px;
    margin: 0.4rem 0 0.4rem auto;
    max-width: 78%;
    width: fit-content;
    font-size: 0.87rem;
    line-height: 1.55;
    display: block;
    word-break: break-word;
}
.bubble-ai {
    background: #ffffff;
    color: #1e293b !important;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #0d9488;
    padding: 0.8rem 1rem;
    border-radius: 3px 14px 14px 14px;
    margin: 0.4rem 0;
    max-width: 100%;
    font-size: 0.86rem;
    line-height: 1.7;
    display: block;
    word-break: break-word;
}
.bubble-ai strong { color: #0f172a; }
.bubble-ai code {
    background: #f1f5f9;
    color: #0f766e !important;
    padding: 1px 5px;
    border-radius: 4px;
    font-size: 0.78rem;
}
.bubble-ai pre {
    background: #f1f5f9;
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 0.73rem;
    overflow-x: auto;
    margin-top: 5px;
    color: #334155;
}

/* ── Tags ── */
.tag  { display:inline-block; padding:1px 7px; border-radius:20px;
        font-size:0.67rem; font-weight:700; margin:1px; }
.ts   { background:#ccfbf1; color:#0f766e; }
.ta   { background:#dbeafe; color:#1d4ed8; }
.tc   { background:#fee2e2; color:#dc2626; }
.tw   { background:#fef9c3; color:#ca8a04; }
.ti   { background:#dbeafe; color:#2563eb; }
.tok  { background:#dcfce7; color:#15803d; }

/* ── Stat cards ── */
.sc-row {
    display: flex;
    gap: 5px;
    flex-wrap: wrap;
    margin: 6px 0;
}
.sc {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 5px 10px;
    min-width: 75px;
    flex: 1 1 auto;
}
.sc-l { font-size: 0.62rem; color: #64748b; margin-bottom: 1px; }
.sc-v { font-size: 0.92rem; font-weight: 700; color: #0f172a; }

/* ── Mini bar ── */
.bar-row {
    display: flex;
    align-items: center;
    gap: 6px;
    margin: 3px 0;
}
.bar-bg {
    flex: 1;
    background: #f1f5f9;
    border-radius: 4px;
    height: 11px;
    overflow: hidden;
}
.bar-fill { height: 100%; border-radius: 4px; }

/* ── Sources strip ── */
.sources {
    background: #f8fafc;
    border: 1px solid #e9eef4;
    border-top: none;
    border-radius: 0 0 8px 8px;
    padding: 5px 10px;
    font-size: 0.68rem;
    color: #94a3b8;
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    align-items: center;
}
.src-chip {
    background: #e0f2fe;
    color: #0369a1;
    border-radius: 10px;
    padding: 1px 7px;
    font-size: 0.65rem;
    font-weight: 600;
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
.src-chip.stats      { background:#fef9c3; color:#854d0e; }

/* ── Streamlit buttons — action buttons ── */
div[data-testid="stButton"] > button {
    width: 100% !important;
    text-align: left !important;
    justify-content: flex-start !important;
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    color: #334155 !important;
    font-size: 0.79rem !important;
    padding: 0.38rem 0.72rem !important;
    transition: all 0.12s ease;
    white-space: normal !important;
    word-break: break-word !important;
    height: auto !important;
    min-height: 34px !important;
    line-height: 1.4 !important;
}
div[data-testid="stButton"] > button:hover {
    border-color: #0d9488 !important;
    background: #f0fdf4 !important;
    color: #0f172a !important;
}

/* ── Send button ── */
.send-wrap div[data-testid="stButton"] > button {
    background: #0d9488 !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    justify-content: center !important;
    font-size: 0.88rem !important;
    text-align: center !important;
}
.send-wrap div[data-testid="stButton"] > button:hover {
    background: #0f766e !important;
}

/* ── Quick chip buttons ── */
.chip-wrap div[data-testid="stButton"] > button {
    border-radius: 20px !important;
    font-size: 0.7rem !important;
    padding: 0.16rem 0.65rem !important;
    background: #f1f5f9 !important;
    color: #475569 !important;
    width: auto !important;
    min-height: 28px !important;
    white-space: nowrap !important;
}
.chip-wrap div[data-testid="stButton"] > button:hover {
    background: #e0f2fe !important;
    border-color: #7dd3fc !important;
    color: #0369a1 !important;
}

/* ── Text area — CRITICAL: visible text ── */
.stTextArea > div > div > textarea {
    border-radius: 10px !important;
    border: 1.5px solid #cbd5e1 !important;
    font-size: 0.88rem !important;
    resize: none !important;
    background: #ffffff !important;
    color: #0f172a !important;
    caret-color: #0d9488 !important;
    padding: 10px 12px !important;
    line-height: 1.5 !important;
}
.stTextArea > div > div > textarea:focus {
    border-color: #0d9488 !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(13,148,136,0.15) !important;
    color: #0f172a !important;
}
.stTextArea > div > div > textarea::placeholder {
    color: #94a3b8 !important;
    opacity: 1 !important;
}

/* ── RAG index bar ── */
.rag-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    padding: 5px 1rem;
    background: #f8fafc;
    border-bottom: 1px solid #e9eef4;
    align-items: center;
}
.rag-label {
    font-size: 0.65rem;
    color: #94a3b8;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .05em;
}

/* ── Welcome screen ── */
.welcome {
    text-align: center;
    padding: 1.8rem 1rem 1rem;
    color: #64748b;
}
.welcome-title {
    font-size: 1.05rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 6px;
}
.welcome-sub {
    font-size: 0.82rem;
    line-height: 1.65;
    max-width: 560px;
    margin: 0 auto 1rem;
    color: #475569;
}
.tip-chip {
    display: inline-block;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 0.74rem;
    margin: 3px;
    color: #475569;
    cursor: default;
}

/* ── Input area background fix ── */
.input-area {
    background: #ffffff;
    border-top: 1px solid #e9eef4;
    padding: 0.65rem 1rem 0.5rem;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #f8fafc; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }

/* ── Responsive fixes ── */
@media (max-width: 640px) {
    .bubble-user, .bubble-ai { max-width: 98%; font-size: 0.83rem; }
    .topbar-right { display: none; }
    .sc { min-width: 60px; }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────── Init ────────────────────────────────────────────
@st.cache_resource(show_spinner="🧬 Building RAG knowledge index…")
def load_rag():
    return ClinicalRAG()

rag   = load_rag()
stats = rag.stats()

CHIPS = [
    ("🛡️ DQ Audit",      "Show all data quality issues and remediations"),
    ("📋 SDTM Tables",   "List all SDTM and ADaM tables"),
    ("🔗 ER Diagram",    "Explain ER relationships between all tables"),
    ("📖 Dictionary",    "Show full data dictionary for all columns"),
    ("📚 Glossary",      "List all CDISC and GCP glossary terms"),
    ("📊 Dashboard",     "Show DQ completeness metrics and error trends"),
    ("📜 Reg Report",    "Show regulatory submission status and dossier"),
    ("⚠️ AE Summary",   "Summarise AE domain data and severity breakdown"),
]

if "messages"  not in st.session_state: st.session_state.messages  = []
if "pending"   not in st.session_state: st.session_state.pending   = ""
if "input_key" not in st.session_state: st.session_state.input_key = 0


# ─────────────────────────── Submit ──────────────────────────────────────────
def submit(q: str):
    q = q.strip()
    if not q:
        return

    # 1. Try stats engine first (highest priority for numeric queries)
    stats_result = answer_stats_query(q)
    if stats_result:
        html, btns = stats_result
        st.session_state.messages.append({"role": "user", "text": q})
        st.session_state.messages.append({
            "role": "ai", "html": html, "btns": btns,
            "sources": [{"category": "stats", "title": f"Dataset Statistics Engine · {q[:40]}",
                         "score": 1.0}],
        })
    else:
        # 2. RAG retrieval + answer engine
        chunks = rag.retrieve(q, top_k=6)
        html, btns = generate_answer(q, chunks)
        st.session_state.messages.append({"role": "user", "text": q})
        st.session_state.messages.append({
            "role": "ai", "html": html, "btns": btns,
            "sources": chunks,
        })

    st.session_state.pending   = ""
    st.session_state.input_key += 1


if st.session_state.pending:
    submit(st.session_state.pending)


# ─────────────────────────── Top bar ─────────────────────────────────────────
st.markdown(f"""
<div class="topbar">
  <span style="font-size:1.2rem">🧬</span>
  <span class="topbar-title">DataGenome AI</span>
  <span class="topbar-sub">RAG · CDISC Clinical Intelligence · No API Key</span>
  <div class="topbar-right">
    <span class="dot"></span>
    STUDY-2026-FIBER &nbsp;·&nbsp; {stats['total']} knowledge chunks
  </div>
</div>
""", unsafe_allow_html=True)

# ── RAG index bar ─────────────────────────────────────────────────────────────
chip_html = "".join(
    f'<span class="src-chip {cat}">{cat} · {n}</span>'
    for cat, n in sorted(stats["by_category"].items())
)
st.markdown(
    f'<div class="rag-bar">'
    f'<span class="rag-label">RAG Index &nbsp;</span>{chip_html}'
    f'<span style="margin-left:auto;font-size:0.65rem;color:#cbd5e1">'
    f'BM25 · stats engine · free · no API key</span></div>',
    unsafe_allow_html=True,
)

# ── Quick chips ───────────────────────────────────────────────────────────────
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
      <div style="font-size:2rem;margin-bottom:0.4rem">🧬</div>
      <div class="welcome-title">DataGenome AI — RAG Clinical Intelligence</div>
      <div class="welcome-sub">
        Ask anything about <strong>STUDY-2026-FIBER</strong> in plain English.<br>
        Schemas · DQ issues · Statistics · ER diagrams · Glossary · Regulatory — all through chat.
      </div>
      <div>
        <span class="tip-chip">💡 Give me the maximum age in the DM table</span>
        <span class="tip-chip">💡 What is the mean ALT in LB?</span>
        <span class="tip-chip">💡 How many subjects are male?</span>
        <span class="tip-chip">💡 Show missing values in LB</span>
        <span class="tip-chip">💡 Distribution of RACE in DM</span>
        <span class="tip-chip">💡 What DQ issues are blocking submission?</span>
        <span class="tip-chip">💡 What is ALCOA+ and why does it matter?</span>
        <span class="tip-chip">💡 Describe the AE table columns</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────── Chat messages ───────────────────────────────────
chat_container = st.container()
with chat_container:
    for idx, msg in enumerate(st.session_state.messages):

        if msg["role"] == "user":
            st.markdown(
                f'<div class="bubble-user">👤 {msg["text"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="bubble-ai">🧬 {msg["html"]}</div>',
                unsafe_allow_html=True,
            )

            # Sources strip
            sources = msg.get("sources", [])
            if sources:
                def _src_chip(s):
                    cat   = s["category"]
                    score = s.get("score", "—")
                    title = s["title"][:45] + ("…" if len(s["title"]) > 45 else "")
                    return f'<span class="src-chip {cat}" title="score: {score}">{title}</span>'
                chips = "".join(_src_chip(s) for s in sources[:5])
                st.markdown(
                    f'<div class="sources">📎 {chips}</div>',
                    unsafe_allow_html=True,
                )

            # Action buttons
            btns = msg.get("btns", [])
            if btns:
                n = min(len(btns), 4)
                bc = st.columns(n)
                for j, (lbl, q) in enumerate(btns):
                    with bc[j % n]:
                        if st.button(lbl, key=f"b_{idx}_{j}"):
                            st.session_state.pending = q
                            st.rerun()

st.markdown("<hr style='margin:0.25rem 0 0;border-color:#e9eef4'>", unsafe_allow_html=True)

# ─────────────────────────── Input row ───────────────────────────────────────
st.markdown('<div class="input-area">', unsafe_allow_html=True)
in_col, btn_col = st.columns([6, 1])
with in_col:
    user_input = st.text_area(
        "Message",
        label_visibility="collapsed",
        placeholder="Ask anything… e.g. 'Give the maximum age in DM' · 'Show DQ issues' · 'What is ARMCD?'",
        height=70,
        key=f"inp_{st.session_state.input_key}",
    )
with btn_col:
    st.markdown("<div class='send-wrap' style='padding-top:4px'>", unsafe_allow_html=True)
    send = st.button("Send ▶", key="send_btn", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

if send and user_input and user_input.strip():
    submit(user_input)
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<div style='text-align:center;font-size:0.65rem;color:#cbd5e1;padding:0.2rem 0 0.4rem'>"
    "BM25 retrieval · Stats engine · 78 clinical chunks · "
    "Zero cost · No API key · GitHub + Streamlit Cloud ready"
    "</div>",
    unsafe_allow_html=True,
)
