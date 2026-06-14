"""
DataGenome AI — Universal RAG Chatbot
Upload CSV/Excel/JSON/SQLite or connect to a database.
Click Analyse → chatbot understands your full dataset structure.
No API key · No cost · Streamlit Cloud ready.
"""
import re, io, sqlite3
import streamlit as st
import pandas as pd

from data_loader import load_uploaded_file, load_db_uri, load_sqlite_path
from analyzer import build_analysis_report
from dynamic_rag import DynamicRAG
from dynamic_answer import generate_dynamic_answer
from stats_engine import answer_stats_query     # static CDISC fallback
from rag_engine import ClinicalRAG              # static CDISC fallback
from answer_engine import generate_answer       # static CDISC fallback

st.set_page_config(
    page_title="DataGenome AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────── CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
*,*::before,*::after{box-sizing:border-box}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:0!important;max-width:100%!important}

/* Top bar */
.topbar{background:linear-gradient(135deg,#0f172a 0%,#134e4a 100%);
  color:white;padding:.5rem 1.2rem;display:flex;align-items:center;gap:.6rem;flex-wrap:wrap}
.topbar-title{font-size:1rem;font-weight:800}
.topbar-sub{font-size:.68rem;opacity:.45}
.topbar-right{margin-left:auto;font-size:.68rem;opacity:.6;display:flex;align-items:center;gap:5px}
.dot{width:7px;height:7px;border-radius:50%;background:#22c55e;display:inline-block}

/* Bubbles */
.bubble-user{background:#0d9488;color:#fff!important;padding:.6rem 1rem;
  border-radius:14px 14px 3px 14px;margin:.4rem 0 .4rem auto;
  max-width:78%;width:fit-content;font-size:.87rem;line-height:1.55;
  display:block;word-break:break-word}
.bubble-ai{background:#fff;color:#1e293b!important;border:1px solid #e2e8f0;
  border-left:3px solid #0d9488;padding:.8rem 1rem;
  border-radius:3px 14px 14px 14px;margin:.4rem 0;
  max-width:100%;font-size:.86rem;line-height:1.7;display:block;word-break:break-word}
.bubble-ai strong{color:#0f172a}
.bubble-ai code{background:#f1f5f9;color:#0f766e!important;
  padding:1px 5px;border-radius:4px;font-size:.78rem}
.bubble-ai pre{background:#f1f5f9;padding:6px 10px;border-radius:6px;
  font-size:.73rem;overflow-x:auto;color:#334155}

/* Tags & chips */
.tag{display:inline-block;padding:1px 7px;border-radius:20px;font-size:.67rem;font-weight:700;margin:1px}
.ts{background:#ccfbf1;color:#0f766e}.ta{background:#dbeafe;color:#1d4ed8}
.tc{background:#fee2e2;color:#dc2626}.tw{background:#fef9c3;color:#ca8a04}

/* Stat cards */
.sc-row{display:flex;gap:5px;flex-wrap:wrap;margin:6px 0}
.sc{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
  padding:5px 10px;min-width:72px;flex:1 1 auto}
.sc-l{font-size:.62rem;color:#64748b;margin-bottom:1px}
.sc-v{font-size:.9rem;font-weight:700;color:#0f172a}

/* Bar */
.bar-row{display:flex;align-items:center;gap:6px;margin:3px 0}
.bar-bg{flex:1;background:#f1f5f9;border-radius:4px;height:11px;overflow:hidden}
.bar-fill{height:100%;border-radius:4px}

/* Sources */
.sources{background:#f8fafc;border:1px solid #e9eef4;border-top:none;
  border-radius:0 0 8px 8px;padding:5px 10px;font-size:.68rem;color:#94a3b8;
  display:flex;flex-wrap:wrap;gap:4px;align-items:center}
.src-chip{background:#e0f2fe;color:#0369a1;border-radius:10px;padding:1px 7px;
  font-size:.65rem;font-weight:600}
.src-chip.dq{background:#fee2e2;color:#dc2626}
.src-chip.schema{background:#ccfbf1;color:#0f766e}
.src-chip.column{background:#ccfbf1;color:#0f766e}
.src-chip.relationship{background:#dbeafe;color:#1d4ed8}
.src-chip.overview{background:#f3e8ff;color:#7c3aed}
.src-chip.stats{background:#fef9c3;color:#854d0e}
.src-chip.glossary{background:#fef9c3;color:#854d0e}
.src-chip.er{background:#dbeafe;color:#1d4ed8}

/* Buttons */
div[data-testid="stButton"]>button{
  width:100%!important;text-align:left!important;justify-content:flex-start!important;
  background:#f8fafc!important;border:1px solid #e2e8f0!important;
  border-radius:8px!important;color:#334155!important;font-size:.79rem!important;
  padding:.38rem .72rem!important;white-space:normal!important;
  word-break:break-word!important;height:auto!important;min-height:32px!important;
  line-height:1.4!important}
div[data-testid="stButton"]>button:hover{
  border-color:#0d9488!important;background:#f0fdf4!important;color:#0f172a!important}

/* Send */
.send-wrap div[data-testid="stButton"]>button{
  background:#0d9488!important;color:#fff!important;border:none!important;
  font-weight:700!important;border-radius:10px!important;
  justify-content:center!important;font-size:.88rem!important;text-align:center!important}
.send-wrap div[data-testid="stButton"]>button:hover{background:#0f766e!important}

/* Analyse button */
.analyse-wrap div[data-testid="stButton"]>button{
  background:#7c3aed!important;color:#fff!important;border:none!important;
  font-weight:700!important;border-radius:10px!important;
  justify-content:center!important;font-size:.88rem!important;text-align:center!important}
.analyse-wrap div[data-testid="stButton"]>button:hover{background:#6d28d9!important}

/* Chips */
.chip-wrap div[data-testid="stButton"]>button{
  border-radius:20px!important;font-size:.7rem!important;
  padding:.16rem .65rem!important;background:#f1f5f9!important;
  color:#475569!important;width:auto!important;min-height:26px!important;white-space:nowrap!important}
.chip-wrap div[data-testid="stButton"]>button:hover{
  background:#e0f2fe!important;border-color:#7dd3fc!important;color:#0369a1!important}

/* Textarea — CRITICAL text visibility */
.stTextArea>div>div>textarea{
  border-radius:10px!important;border:1.5px solid #cbd5e1!important;
  font-size:.88rem!important;resize:none!important;
  background:#ffffff!important;color:#0f172a!important;
  caret-color:#0d9488!important;padding:10px 12px!important;line-height:1.5!important}
.stTextArea>div>div>textarea:focus{
  border-color:#0d9488!important;outline:none!important;
  box-shadow:0 0 0 2px rgba(13,148,136,.15)!important;color:#0f172a!important}
.stTextArea>div>div>textarea::placeholder{color:#94a3b8!important;opacity:1!important}

/* Text input */
.stTextInput>div>div>input{
  border-radius:8px!important;border:1.5px solid #cbd5e1!important;
  font-size:.85rem!important;background:#fff!important;
  color:#0f172a!important;padding:6px 10px!important}
.stTextInput>div>div>input:focus{border-color:#0d9488!important;color:#0f172a!important}

/* Sidebar */
[data-testid="stSidebar"]{background:#f8fafc;border-right:1px solid #e2e8f0}
[data-testid="stSidebar"] .stMarkdown p{font-size:.82rem;color:#475569}

/* Upload zone */
.upload-zone{border:2px dashed #cbd5e1;border-radius:12px;
  padding:1.2rem;text-align:center;background:#fafafa;margin:.5rem 0}
.upload-zone:hover{border-color:#0d9488;background:#f0fdf4}

/* Status badges */
.status-ok{background:#dcfce7;color:#15803d;border-radius:20px;
  padding:2px 9px;font-size:.68rem;font-weight:700}
.status-warn{background:#fef9c3;color:#854d0e;border-radius:20px;
  padding:2px 9px;font-size:.68rem;font-weight:700}
.status-err{background:#fee2e2;color:#dc2626;border-radius:20px;
  padding:2px 9px;font-size:.68rem;font-weight:700}

/* Welcome */
.welcome{text-align:center;padding:2rem 1rem 1rem;color:#64748b}
.welcome-title{font-size:1rem;font-weight:800;color:#0f172a;margin-bottom:6px}
.tip-chip{display:inline-block;background:#f1f5f9;border:1px solid #e2e8f0;
  border-radius:8px;padding:4px 10px;font-size:.74rem;margin:3px;color:#475569}

::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:#f8fafc}
::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:3px}

@media(max-width:640px){
  .bubble-user,.bubble-ai{max-width:98%;font-size:.83rem}
  .topbar-right{display:none}
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────── Caches ───────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_static_rag():
    return ClinicalRAG()

# ─────────────────────── Session state ───────────────────────────────────────
def _init():
    defaults = {
        "messages":     [],
        "pending":      "",
        "input_key":    0,
        "datasets":     [],     # list[DatasetInfo]
        "report":       None,   # analysis report dict
        "dyn_rag":      None,   # DynamicRAG instance
        "mode":         "static",  # "static" | "dynamic"
        "db_connected": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()

static_rag = load_static_rag()

# ─────────────────────── Submit ───────────────────────────────────────────────
def submit(q: str):
    q = q.strip()
    if not q:
        return

    if st.session_state.mode == "dynamic" and st.session_state.report:
        # Dynamic mode — user uploaded datasets
        dyn   = st.session_state.dyn_rag
        chunks = dyn.retrieve(q, top_k=6)
        html, btns = generate_dynamic_answer(q, chunks, st.session_state.report)
        sources = chunks
    else:
        # Static CDISC mode — try stats engine first
        stat_res = answer_stats_query(q)
        if stat_res:
            html, btns = stat_res
            sources = [{"category":"stats","title":"Stats Engine","score":1.0}]
        else:
            chunks   = static_rag.retrieve(q, top_k=6)
            html, btns = generate_answer(q, chunks)
            sources  = chunks

    st.session_state.messages.append({"role": "user", "text": q})
    st.session_state.messages.append({
        "role": "ai", "html": html, "btns": btns, "sources": sources
    })
    st.session_state.pending   = ""
    st.session_state.input_key += 1

if st.session_state.pending:
    submit(st.session_state.pending)

# ─────────────────────── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧬 DataGenome AI")
    st.markdown("---")

    # ── Mode selector
    mode_choice = st.radio(
        "Data source",
        ["📁 Upload files", "🔌 Connect database", "🧬 Built-in CDISC demo"],
        key="mode_radio",
        label_visibility="collapsed",
    )

    st.markdown("---")

    # ══════════════════════════════════════════════════════
    # FILE UPLOAD MODE
    # ══════════════════════════════════════════════════════
    if mode_choice == "📁 Upload files":
        st.markdown("**Upload datasets**")
        st.caption("CSV · TSV · TXT · Excel · JSON · SQLite  (multiple files OK)")

        uploaded = st.file_uploader(
            "Drop files here",
            type=["csv","tsv","txt","xlsx","xls","json","db","sqlite","sqlite3"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="file_uploader",
        )

        if uploaded:
            st.markdown(f"**{len(uploaded)} file(s) selected:**")
            for f in uploaded:
                kb = round(len(f.getvalue()) / 1024, 1)
                st.markdown(f'<span class="status-ok">✓</span> `{f.name}` ({kb} KB)',
                            unsafe_allow_html=True)

            st.markdown("")
            st.markdown('<div class="analyse-wrap">', unsafe_allow_html=True)
            analyse_clicked = st.button("🔍 Analyse Datasets", key="analyse_btn",
                                         use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if analyse_clicked:
                with st.spinner("📊 Loading & profiling datasets…"):
                    all_ds = []
                    errors = []
                    for f in uploaded:
                        try:
                            f.seek(0)
                            ds_list = load_uploaded_file(f)
                            all_ds.extend(ds_list)
                        except Exception as e:
                            errors.append(f"{f.name}: {e}")

                    if errors:
                        for err in errors:
                            st.error(err)

                    if all_ds:
                        with st.spinner("🔗 Detecting relationships & DQ issues…"):
                            report = build_analysis_report(all_ds)
                            dyn_rag = DynamicRAG(report)

                        st.session_state.datasets  = all_ds
                        st.session_state.report    = report
                        st.session_state.dyn_rag   = dyn_rag
                        st.session_state.mode      = "dynamic"
                        st.session_state.messages  = []
                        st.session_state.input_key += 1

                        s = report["summary"]
                        # Post welcome message
                        tnames = ", ".join(f"`{d.name}`" for d in all_ds)
                        html = (
                            f"<strong>✅ Analysis complete — {len(all_ds)} dataset(s) loaded</strong><br>"
                            f"<span style='font-size:.8rem;color:#64748b'>"
                            f"Tables: {tnames}</span>"
                            f"<div class='sc-row' style='margin-top:8px'>"
                            + "".join(
                                f"<div class='sc'><div class='sc-l'>{lbl}</div>"
                                f"<div class='sc-v' style='color:{col}'>{val}</div></div>"
                                for lbl,val,col in [
                                    ("Datasets",      len(all_ds),              "#0f172a"),
                                    ("Total rows",    f"{s['total_rows']:,}",   "#0f172a"),
                                    ("Total cols",    s["total_cols"],           "#0f172a"),
                                    ("Relationships", s["total_relationships"], "#0d9488"),
                                    ("DQ issues",     s["total_dq_issues"],
                                     "#dc2626" if s["critical_dq"]>0 else "#ca8a04"),
                                ]
                            )
                            + "</div>"
                        )
                        btns = (
                            [("📊 Full overview",       "Show dataset overview"),
                             ("🔗 Relationships",       "Show all relationships between datasets"),
                             ("🛡️ DQ issues",           "Show all data quality issues")]
                            + [(f"📋 {d.name} schema",  f"Show schema of {d.name}") for d in all_ds[:4]]
                            + [(f"📊 {d.name} stats",   f"Show statistics for {d.name}") for d in all_ds[:4]]
                        )
                        st.session_state.messages.append({
                            "role":"ai","html":html,"btns":btns,"sources":[]
                        })
                        st.rerun()

    # ══════════════════════════════════════════════════════
    # DATABASE CONNECT MODE
    # ══════════════════════════════════════════════════════
    elif mode_choice == "🔌 Connect database":
        st.markdown("**Database connection**")

        db_type = st.selectbox("Type", ["SQLite (file path)",
                                         "PostgreSQL", "MySQL"], key="db_type")
        if db_type == "SQLite (file path)":
            db_path = st.text_input("SQLite file path",
                                     placeholder="/path/to/database.db", key="sqlite_path")
            conn_uri = f"sqlite:///{db_path}" if db_path else ""
        elif db_type == "PostgreSQL":
            col1, col2 = st.columns(2)
            with col1: host = st.text_input("Host", value="localhost", key="pg_host")
            with col2: port = st.text_input("Port", value="5432", key="pg_port")
            db   = st.text_input("Database", key="pg_db")
            user = st.text_input("User",     key="pg_user")
            pwd  = st.text_input("Password", type="password", key="pg_pwd")
            conn_uri = f"postgresql://{user}:{pwd}@{host}:{port}/{db}" if db else ""
        else:
            col1, col2 = st.columns(2)
            with col1: host = st.text_input("Host", value="localhost", key="my_host")
            with col2: port = st.text_input("Port", value="3306",     key="my_port")
            db   = st.text_input("Database", key="my_db")
            user = st.text_input("User",     key="my_user")
            pwd  = st.text_input("Password", type="password", key="my_pwd")
            conn_uri = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}" if db else ""

        if conn_uri:
            st.markdown('<div class="analyse-wrap">', unsafe_allow_html=True)
            if st.button("🔌 Connect & Analyse", key="db_connect_btn",
                         use_container_width=True):
                with st.spinner("Connecting…"):
                    try:
                        if db_type == "SQLite (file path)":
                            all_ds = load_sqlite_path(conn_uri.replace("sqlite:///",""))
                        else:
                            all_ds = load_db_uri(conn_uri)

                        report  = build_analysis_report(all_ds)
                        dyn_rag = DynamicRAG(report)

                        st.session_state.datasets     = all_ds
                        st.session_state.report       = report
                        st.session_state.dyn_rag      = dyn_rag
                        st.session_state.mode         = "dynamic"
                        st.session_state.db_connected = True
                        st.session_state.messages     = []

                        s = report["summary"]
                        html = (
                            f"<strong>✅ DB connected — {len(all_ds)} table(s) loaded</strong>"
                            f"<div class='sc-row' style='margin-top:8px'>"
                            + "".join(
                                f"<div class='sc'><div class='sc-l'>{lbl}</div>"
                                f"<div class='sc-v'>{val}</div></div>"
                                for lbl,val in [
                                    ("Tables",    len(all_ds)),
                                    ("Rows",      f"{s['total_rows']:,}"),
                                    ("Relations", s["total_relationships"]),
                                    ("DQ issues", s["total_dq_issues"]),
                                ]
                            )
                            + "</div>"
                        )
                        btns = (
                            [("📊 Overview",      "Show dataset overview"),
                             ("🔗 Relationships", "Show all relationships between datasets"),
                             ("🛡️ DQ Issues",    "Show all data quality issues")]
                            + [(f"📋 {d.name}", f"Show schema of {d.name}") for d in all_ds[:5]]
                        )
                        st.session_state.messages.append({
                            "role":"ai","html":html,"btns":btns,"sources":[]
                        })
                        st.session_state.input_key += 1
                        st.rerun()
                    except Exception as e:
                        st.error(f"Connection failed: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    # STATIC CDISC DEMO MODE
    # ══════════════════════════════════════════════════════
    else:
        st.markdown("**Built-in CDISC demo**")
        st.caption("STUDY-2026-FIBER · 5 tables · 10,240 records")
        if st.session_state.mode != "static":
            if st.button("Use CDISC demo", key="use_static"):
                st.session_state.mode     = "static"
                st.session_state.messages = []
                st.session_state.input_key += 1
                st.rerun()

    st.markdown("---")

    # Current state summary
    if st.session_state.mode == "dynamic" and st.session_state.report:
        s  = st.session_state.report["summary"]
        ds = st.session_state.datasets
        st.markdown("**📂 Loaded datasets**")
        for d in ds:
            st.markdown(
                f'<span class="status-ok">✓</span> `{d.name}` '
                f'({d.row_count:,} rows · {d.col_count} cols)',
                unsafe_allow_html=True,
            )
        st.caption(
            f"Relationships: {s['total_relationships']} · "
            f"DQ issues: {s['total_dq_issues']}"
        )
        if st.button("🗑️ Clear & reset", key="clear_btn"):
            for k in ["datasets","report","dyn_rag","messages"]:
                st.session_state[k] = [] if k in ("datasets","messages") else None
            st.session_state.mode      = "static"
            st.session_state.input_key += 1
            st.rerun()
    elif st.session_state.mode == "static":
        st.markdown(
            '<span class="status-ok">✓</span> CDISC demo active',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.caption("BM25 retrieval · Free · No API key\nStreamlit Cloud compatible")

# ─────────────────────── Top bar ──────────────────────────────────────────────
mode_label = "User data" if st.session_state.mode=="dynamic" else "CDISC demo"
chunk_count = (st.session_state.dyn_rag.stats()["total"]
               if st.session_state.dyn_rag else static_rag.stats()["total"])
st.markdown(f"""
<div class="topbar">
  <span style="font-size:1.2rem">🧬</span>
  <span class="topbar-title">DataGenome AI</span>
  <span class="topbar-sub">Upload · Analyse · Chat — No API key</span>
  <div class="topbar-right">
    <span class="dot"></span>
    {mode_label} &nbsp;·&nbsp; {chunk_count} knowledge chunks
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────── Quick chips ─────────────────────────────────────────
if st.session_state.mode == "dynamic" and st.session_state.report:
    ds_list = st.session_state.datasets
    CHIPS = (
        [("📊 Overview",      "Show dataset overview"),
         ("🔗 Relationships", "Show all relationships between datasets"),
         ("🛡️ DQ Issues",    "Show all data quality issues")]
        + [(f"📋 {d.name}",   f"Show schema of {d.name}") for d in ds_list[:5]]
    )
else:
    CHIPS = [
        ("🛡️ DQ Audit",     "Show all data quality issues and remediations"),
        ("📋 SDTM Tables",  "List all SDTM and ADaM tables"),
        ("🔗 ER Diagram",   "Explain ER relationships between all tables"),
        ("📖 Dictionary",   "Show full data dictionary for all columns"),
        ("📚 Glossary",     "List all CDISC and GCP glossary terms"),
        ("📊 DQ Dashboard", "Show DQ completeness metrics and error trends"),
        ("📜 Reg Report",   "Show regulatory submission status and dossier"),
    ]

chip_cols = st.columns(min(len(CHIPS), 8))
for col, (label, query) in zip(chip_cols, CHIPS[:8]):
    with col:
        st.markdown('<div class="chip-wrap">', unsafe_allow_html=True)
        if st.button(label, key=f"chip_{label}"):
            st.session_state.pending = query
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr style='margin:0;border-color:#e9eef4'>", unsafe_allow_html=True)

# ─────────────────────── Welcome screen ───────────────────────────────────────
if not st.session_state.messages:
    if st.session_state.mode == "dynamic":
        pass  # analysis result will be first message
    else:
        st.markdown("""
        <div class="welcome">
          <div style="font-size:2rem;margin-bottom:.4rem">🧬</div>
          <div class="welcome-title">DataGenome AI — Upload, Analyse, Chat</div>
          <div style="font-size:.82rem;line-height:1.65;max-width:580px;
               margin:0 auto .8rem;color:#475569">
            <strong>New:</strong> Upload your own CSV, Excel, JSON or SQLite files in the sidebar,
            click <strong style="color:#7c3aed">🔍 Analyse</strong> — then ask anything about your data.<br>
            Or use the built-in CDISC demo below.
          </div>
          <div>
            <span class="tip-chip">💡 Upload a CSV → click Analyse → ask "What is the max age?"</span>
            <span class="tip-chip">💡 Upload multiple files → ask "Show relationships"</span>
            <span class="tip-chip">💡 Connect SQLite → ask "Show schema of orders table"</span>
            <span class="tip-chip">💡 CDISC demo: "What DQ issues are blocking submission?"</span>
            <span class="tip-chip">💡 CDISC demo: "What is the mean ALT in LB?"</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────── Chat ─────────────────────────────────────────────────
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
        sources = msg.get("sources", [])
        if sources:
            def _chip(s):
                cat   = s.get("category","info")
                score = s.get("score","—")
                title = s.get("title","")
                short = title[:42] + ("…" if len(title)>42 else "")
                return f'<span class="src-chip {cat}" title="score:{score}">{short}</span>'
            chips = "".join(_chip(s) for s in sources[:5])
            st.markdown(f'<div class="sources">📎 {chips}</div>',
                        unsafe_allow_html=True)

        btns = msg.get("btns", [])
        if btns:
            n  = min(len(btns), 4)
            bc = st.columns(n)
            for j, (lbl, q) in enumerate(btns):
                with bc[j % n]:
                    if st.button(lbl, key=f"b_{idx}_{j}"):
                        st.session_state.pending = q
                        st.rerun()

st.markdown("<hr style='margin:.25rem 0 0;border-color:#e9eef4'>", unsafe_allow_html=True)

# ─────────────────────── Input row ────────────────────────────────────────────
in_col, btn_col = st.columns([6, 1])
with in_col:
    placeholder = (
        "Ask about your data… e.g. 'Show schema of sales', "
        "'What is the max revenue?', 'Show missing values'"
        if st.session_state.mode == "dynamic"
        else "Ask about CDISC schemas, DQ issues, glossary terms…"
    )
    user_input = st.text_area(
        "Message",
        label_visibility="collapsed",
        placeholder=placeholder,
        height=68,
        key=f"inp_{st.session_state.input_key}",
    )
with btn_col:
    st.markdown("<div class='send-wrap' style='padding-top:4px'>", unsafe_allow_html=True)
    send = st.button("Send ▶", key="send_btn", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

if send and user_input and user_input.strip():
    submit(user_input)
    st.rerun()

st.markdown(
    "<div style='text-align:center;font-size:.65rem;color:#cbd5e1;padding:.2rem 0 .4rem'>"
    "BM25 retrieval · pandas profiling · free · no API key · GitHub + Streamlit Cloud ready"
    "</div>",
    unsafe_allow_html=True,
)
