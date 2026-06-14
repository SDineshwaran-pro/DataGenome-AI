"""
DataGenome AI — Universal RAG Chatbot
Upload panel always visible at top of page (no sidebar needed).
No API key · No cost · Streamlit Cloud ready.
"""
import re, io, os
import streamlit as st
import pandas as pd

from data_loader import load_uploaded_file, load_db_uri, load_sqlite_path
from analyzer import build_analysis_report
from dynamic_rag import DynamicRAG
from dynamic_answer import generate_dynamic_answer
from stats_engine import answer_stats_query
from rag_engine import ClinicalRAG
from answer_engine import generate_answer

st.set_page_config(
    page_title="DataGenome AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",   # sidebar collapsed — everything on main page
)

# ─────────────────────────── CSS ─────────────────────────────────────────────
st.markdown("""
<style>
*,*::before,*::after{box-sizing:border-box}
#MainMenu,footer,header{visibility:hidden}
[data-testid="collapsedControl"]{display:none}   /* hide sidebar toggle arrow */
.block-container{padding:0!important;max-width:100%!important}

/* ── Top bar ── */
.topbar{background:linear-gradient(135deg,#0f172a 0%,#134e4a 100%);
  color:white;padding:.5rem 1.2rem;display:flex;align-items:center;
  gap:.6rem;flex-wrap:wrap}
.topbar-title{font-size:1rem;font-weight:800}
.topbar-sub{font-size:.68rem;opacity:.45}
.topbar-right{margin-left:auto;font-size:.68rem;opacity:.6;
  display:flex;align-items:center;gap:5px}
.dot{width:7px;height:7px;border-radius:50%;background:#22c55e;display:inline-block}

/* ── Upload panel ── */
.upload-panel{background:#fff;border:1px solid #e2e8f0;border-radius:12px;
  padding:1rem 1.2rem;margin:.6rem 1rem .3rem}
.upload-panel-title{font-size:.82rem;font-weight:700;color:#0f172a;margin-bottom:.5rem;
  display:flex;align-items:center;gap:6px}
.tab-bar{display:flex;gap:6px;margin-bottom:.75rem;flex-wrap:wrap}
.tab-btn{padding:4px 14px;border-radius:20px;font-size:.75rem;font-weight:600;
  cursor:pointer;border:1.5px solid #e2e8f0;background:#f8fafc;color:#475569;
  transition:all .15s}
.tab-btn.active{background:#0d9488;color:white;border-color:#0d9488}
.dataset-pill{display:inline-flex;align-items:center;gap:5px;background:#f0fdf4;
  border:1px solid #86efac;border-radius:20px;padding:3px 10px;
  font-size:.72rem;color:#15803d;font-weight:600;margin:2px}
.analysis-ready{background:#f0fdf4;border:1px solid #86efac;border-radius:10px;
  padding:.5rem .8rem;font-size:.78rem;color:#15803d;
  display:flex;align-items:center;gap:6px}
.analysis-pending{background:#fff7ed;border:1px solid #fed7aa;border-radius:10px;
  padding:.5rem .8rem;font-size:.78rem;color:#c2410c;
  display:flex;align-items:center;gap:6px}

/* ── Bubbles ── */
.bubble-user{background:#0d9488;color:#fff!important;padding:.6rem 1rem;
  border-radius:14px 14px 3px 14px;margin:.4rem 0 .4rem auto;
  max-width:78%;width:fit-content;font-size:.87rem;line-height:1.55;
  display:block;word-break:break-word}
.bubble-ai{background:#fff;color:#1e293b!important;border:1px solid #e2e8f0;
  border-left:3px solid #0d9488;padding:.8rem 1rem;
  border-radius:3px 14px 14px 14px;margin:.4rem 0;
  max-width:100%;font-size:.86rem;line-height:1.7;
  display:block;word-break:break-word}
.bubble-ai strong{color:#0f172a}
.bubble-ai code{background:#f1f5f9;color:#0f766e!important;
  padding:1px 5px;border-radius:4px;font-size:.78rem}
.bubble-ai pre{background:#f1f5f9;padding:6px 10px;border-radius:6px;
  font-size:.73rem;overflow-x:auto;color:#334155}

/* ── Stat cards ── */
.sc-row{display:flex;gap:5px;flex-wrap:wrap;margin:6px 0}
.sc{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
  padding:5px 10px;min-width:72px;flex:1 1 auto}
.sc-l{font-size:.62rem;color:#64748b;margin-bottom:1px}
.sc-v{font-size:.9rem;font-weight:700;color:#0f172a}

/* ── Bar ── */
.bar-row{display:flex;align-items:center;gap:6px;margin:3px 0}
.bar-bg{flex:1;background:#f1f5f9;border-radius:4px;height:11px;overflow:hidden}
.bar-fill{height:100%;border-radius:4px}

/* ── Sources ── */
.sources{background:#f8fafc;border:1px solid #e9eef4;border-top:none;
  border-radius:0 0 8px 8px;padding:5px 10px;font-size:.68rem;color:#94a3b8;
  display:flex;flex-wrap:wrap;gap:4px;align-items:center}
.src-chip{background:#e0f2fe;color:#0369a1;border-radius:10px;
  padding:1px 7px;font-size:.65rem;font-weight:600}
.src-chip.dq{background:#fee2e2;color:#dc2626}
.src-chip.schema,.src-chip.column{background:#ccfbf1;color:#0f766e}
.src-chip.relationship{background:#dbeafe;color:#1d4ed8}
.src-chip.overview{background:#f3e8ff;color:#7c3aed}
.src-chip.stats,.src-chip.glossary{background:#fef9c3;color:#854d0e}
.src-chip.er{background:#dbeafe;color:#1d4ed8}

/* ── Buttons ── */
div[data-testid="stButton"]>button{
  width:100%!important;text-align:left!important;
  justify-content:flex-start!important;background:#f8fafc!important;
  border:1px solid #e2e8f0!important;border-radius:8px!important;
  color:#334155!important;font-size:.79rem!important;
  padding:.38rem .72rem!important;white-space:normal!important;
  word-break:break-word!important;height:auto!important;
  min-height:32px!important;line-height:1.4!important}
div[data-testid="stButton"]>button:hover{
  border-color:#0d9488!important;background:#f0fdf4!important;color:#0f172a!important}

/* Send */
.send-wrap div[data-testid="stButton"]>button{
  background:#0d9488!important;color:#fff!important;border:none!important;
  font-weight:700!important;border-radius:10px!important;
  justify-content:center!important;font-size:.88rem!important}
.send-wrap div[data-testid="stButton"]>button:hover{background:#0f766e!important}

/* Analyse */
.analyse-wrap div[data-testid="stButton"]>button{
  background:#7c3aed!important;color:#fff!important;border:none!important;
  font-weight:700!important;border-radius:10px!important;
  justify-content:center!important;font-size:.85rem!important}
.analyse-wrap div[data-testid="stButton"]>button:hover{background:#6d28d9!important}

/* Clear */
.clear-wrap div[data-testid="stButton"]>button{
  background:#fee2e2!important;color:#dc2626!important;
  border:1px solid #fca5a5!important;font-weight:600!important;
  border-radius:8px!important;justify-content:center!important;font-size:.78rem!important}

/* Chips */
.chip-wrap div[data-testid="stButton"]>button{
  border-radius:20px!important;font-size:.7rem!important;
  padding:.16rem .65rem!important;background:#f1f5f9!important;
  color:#475569!important;width:auto!important;
  min-height:26px!important;white-space:nowrap!important}
.chip-wrap div[data-testid="stButton"]>button:hover{
  background:#e0f2fe!important;border-color:#7dd3fc!important;color:#0369a1!important}

/* Textarea — text MUST be visible */
.stTextArea>div>div>textarea{
  border-radius:10px!important;border:1.5px solid #cbd5e1!important;
  font-size:.88rem!important;resize:none!important;
  background:#ffffff!important;color:#0f172a!important;
  caret-color:#0d9488!important;padding:10px 12px!important;line-height:1.5!important}
.stTextArea>div>div>textarea:focus{
  border-color:#0d9488!important;outline:none!important;
  box-shadow:0 0 0 2px rgba(13,148,136,.15)!important;color:#0f172a!important}
.stTextArea>div>div>textarea::placeholder{color:#94a3b8!important;opacity:1!important}

/* Text inputs */
.stTextInput>div>div>input{
  border-radius:8px!important;border:1.5px solid #cbd5e1!important;
  font-size:.85rem!important;background:#fff!important;
  color:#0f172a!important;padding:6px 10px!important}
.stTextInput>div>div>input:focus{border-color:#0d9488!important;color:#0f172a!important}

/* File uploader */
[data-testid="stFileUploader"]{border:2px dashed #cbd5e1;border-radius:10px;
  padding:.5rem;background:#fafafa}
[data-testid="stFileUploader"]:hover{border-color:#0d9488;background:#f0fdf4}
[data-testid="stFileUploaderDropzone"]{background:transparent!important}

/* Selectbox */
.stSelectbox>div>div{border-radius:8px!important;border:1.5px solid #cbd5e1!important;
  background:#fff!important;color:#0f172a!important}

::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:#f8fafc}
::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:3px}

@media(max-width:640px){
  .bubble-user,.bubble-ai{max-width:98%;font-size:.83rem}
  .topbar-right{display:none}
  .sc{min-width:60px}
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────── Init ────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_static_rag():
    return ClinicalRAG()

def _init():
    defaults = {
        "messages":   [],
        "pending":    "",
        "input_key":  0,
        "datasets":   [],
        "report":     None,
        "dyn_rag":    None,
        "mode":       "static",
        "panel_tab":  "files",   # "files" | "database" | "demo"
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()
static_rag = load_static_rag()


# ─────────────────────────── Submit ──────────────────────────────────────────
def submit(q: str):
    q = q.strip()
    if not q:
        return
    if st.session_state.mode == "dynamic" and st.session_state.report:
        dyn    = st.session_state.dyn_rag
        chunks = dyn.retrieve(q, top_k=6)
        html, btns = generate_dynamic_answer(q, chunks, st.session_state.report)
        sources = chunks
    else:
        stat_res = answer_stats_query(q)
        if stat_res:
            html, btns = stat_res
            sources = [{"category":"stats","title":"Stats Engine","score":1.0}]
        else:
            chunks = static_rag.retrieve(q, top_k=6)
            html, btns = generate_answer(q, chunks)
            sources = chunks
    st.session_state.messages.append({"role":"user","text":q})
    st.session_state.messages.append({"role":"ai","html":html,"btns":btns,"sources":sources})
    st.session_state.pending   = ""
    st.session_state.input_key += 1

if st.session_state.pending:
    submit(st.session_state.pending)


# ─────────────────────────── Top bar ─────────────────────────────────────────
mode_lbl = "Your data" if st.session_state.mode == "dynamic" else "CDISC demo"
n_chunks = (st.session_state.dyn_rag.stats()["total"]
            if st.session_state.dyn_rag else static_rag.stats()["total"])

st.markdown(f"""
<div class="topbar">
  <span style="font-size:1.2rem">🧬</span>
  <span class="topbar-title">DataGenome AI</span>
  <span class="topbar-sub">Upload · Analyse · Chat — No API key · Free</span>
  <div class="topbar-right">
    <span class="dot"></span>
    {mode_lbl} &nbsp;·&nbsp; {n_chunks} knowledge chunks
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# UPLOAD / CONNECT PANEL  (always visible, top of page)
# ═══════════════════════════════════════════════════════════════════════════════
with st.container():
    st.markdown('<div class="upload-panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="upload-panel-title">'
        '📂 <span>Data Source</span>'
        '<span style="margin-left:auto;font-size:.7rem;color:#94a3b8;font-weight:400">'
        'Upload files or connect a database, then click Analyse</span></div>',
        unsafe_allow_html=True,
    )

    # Tab selector
    tab_col1, tab_col2, tab_col3 = st.columns(3)
    with tab_col1:
        if st.button("📁 Upload Files",   key="tab_files",
                     type="primary" if st.session_state.panel_tab=="files" else "secondary"):
            st.session_state.panel_tab = "files"
            st.rerun()
    with tab_col2:
        if st.button("🔌 Connect Database", key="tab_db",
                     type="primary" if st.session_state.panel_tab=="database" else "secondary"):
            st.session_state.panel_tab = "database"
            st.rerun()
    with tab_col3:
        if st.button("🧬 CDISC Demo",     key="tab_demo",
                     type="primary" if st.session_state.panel_tab=="demo" else "secondary"):
            st.session_state.panel_tab = "demo"
            st.rerun()

    st.markdown("<hr style='margin:.5rem 0;border-color:#f1f5f9'>", unsafe_allow_html=True)

    # ── TAB: File upload ──────────────────────────────────────────────────────
    if st.session_state.panel_tab == "files":
        up_col, btn_col = st.columns([4, 1])
        with up_col:
            uploaded = st.file_uploader(
                "Drop CSV, Excel, JSON or SQLite files here — multiple files supported",
                type=["csv","tsv","txt","xlsx","xls","json","db","sqlite","sqlite3"],
                accept_multiple_files=True,
                key="file_uploader",
                label_visibility="visible",
            )
        with btn_col:
            st.markdown("<div style='padding-top:1.6rem'>", unsafe_allow_html=True)

            # Status
            if st.session_state.mode == "dynamic" and st.session_state.report:
                ds_list = st.session_state.datasets
                s       = st.session_state.report["summary"]
                pills   = "".join(
                    f'<span class="dataset-pill">✓ {d.name} ({d.row_count:,}r)</span>'
                    for d in ds_list
                )
                st.markdown(
                    f'<div class="analysis-ready">✅ Analysis ready</div>'
                    f'<div style="margin-top:4px;font-size:.72rem">{pills}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='margin-top:6px'>", unsafe_allow_html=True)
                st.markdown('<div class="clear-wrap">', unsafe_allow_html=True)
                if st.button("🗑️ Clear data", key="clear_btn"):
                    for k in ["datasets","report","dyn_rag","messages"]:
                        st.session_state[k] = [] if k in ("datasets","messages") else None
                    st.session_state.mode      = "static"
                    st.session_state.input_key += 1
                    st.rerun()
                st.markdown("</div></div>", unsafe_allow_html=True)

            elif uploaded:
                st.markdown(
                    f'<div class="analysis-pending">'
                    f'⏳ {len(uploaded)} file(s) ready — click Analyse</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='margin-top:6px'>", unsafe_allow_html=True)
                st.markdown('<div class="analyse-wrap">', unsafe_allow_html=True)
                if st.button("🔍 Analyse", key="analyse_btn", use_container_width=True):
                    with st.spinner("📊 Profiling datasets…"):
                        all_ds, errors = [], []
                        for f in uploaded:
                            try:
                                f.seek(0)
                                all_ds.extend(load_uploaded_file(f))
                            except Exception as e:
                                errors.append(f"{f.name}: {e}")
                    if errors:
                        for err in errors:
                            st.error(err)
                    if all_ds:
                        with st.spinner("🔗 Detecting relationships…"):
                            report  = build_analysis_report(all_ds)
                            dyn_rag = DynamicRAG(report)
                        st.session_state.datasets   = all_ds
                        st.session_state.report     = report
                        st.session_state.dyn_rag    = dyn_rag
                        st.session_state.mode       = "dynamic"
                        st.session_state.messages   = []
                        st.session_state.input_key += 1
                        s = report["summary"]
                        tnames = ", ".join(f"`{d.name}`" for d in all_ds)
                        html = (
                            f"<strong>✅ {len(all_ds)} dataset(s) analysed — ready to chat!</strong><br>"
                            f"<span style='font-size:.79rem;color:#64748b'>Tables: {tnames}</span>"
                            f"<div class='sc-row' style='margin-top:8px'>"
                            + "".join(
                                f"<div class='sc'><div class='sc-l'>{lb}</div>"
                                f"<div class='sc-v' style='color:{cl}'>{vl}</div></div>"
                                for lb,vl,cl in [
                                    ("Datasets",     len(all_ds),              "#0f172a"),
                                    ("Total rows",   f"{s['total_rows']:,}",   "#0f172a"),
                                    ("Total cols",   s["total_cols"],           "#0f172a"),
                                    ("Relationships",s["total_relationships"], "#0d9488"),
                                    ("DQ issues",    s["total_dq_issues"],
                                     "#dc2626" if s["critical_dq"]>0 else "#ca8a04"),
                                ]
                            )
                            + "</div>"
                        )
                        btns = (
                            [("📊 Full overview",    "Show dataset overview"),
                             ("🔗 Relationships",    "Show all relationships between datasets"),
                             ("🛡️ DQ Issues",       "Show all data quality issues")]
                            + [(f"📋 {d.name} schema", f"Show schema of {d.name}") for d in all_ds[:4]]
                            + [(f"📊 {d.name} stats",  f"Show statistics for {d.name}") for d in all_ds[:4]]
                        )
                        st.session_state.messages.append(
                            {"role":"ai","html":html,"btns":btns,"sources":[]}
                        )
                        st.rerun()
                st.markdown("</div></div>", unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="font-size:.75rem;color:#94a3b8;padding-top:.5rem">'
                    '👈 Select files to begin</div>',
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB: Database connect ─────────────────────────────────────────────────
    elif st.session_state.panel_tab == "database":
        db_col1, db_col2, db_col3 = st.columns([1, 3, 1])
        with db_col1:
            db_type = st.selectbox(
                "Database type",
                ["SQLite", "PostgreSQL", "MySQL"],
                key="db_type_sel",
                label_visibility="visible",
            )
        with db_col2:
            if db_type == "SQLite":
                db_path = st.text_input(
                    "SQLite file path",
                    placeholder="/path/to/your/database.db  or  ./mydata.sqlite",
                    key="sqlite_path_input",
                )
                conn_uri = db_path.strip() if db_path else ""
            elif db_type == "PostgreSQL":
                c1,c2,c3,c4,c5 = st.columns([2,1,2,1,1])
                with c1: host = st.text_input("Host", value="localhost", key="pg_h")
                with c2: port = st.text_input("Port", value="5432",      key="pg_p")
                with c3: db   = st.text_input("Database",                key="pg_d")
                with c4: user = st.text_input("User",                    key="pg_u")
                with c5: pwd  = st.text_input("Password", type="password", key="pg_pw")
                conn_uri = f"postgresql://{user}:{pwd}@{host}:{port}/{db}" if db and user else ""
            else:
                c1,c2,c3,c4,c5 = st.columns([2,1,2,1,1])
                with c1: host = st.text_input("Host", value="localhost", key="my_h")
                with c2: port = st.text_input("Port", value="3306",      key="my_p")
                with c3: db   = st.text_input("Database",                key="my_d")
                with c4: user = st.text_input("User",                    key="my_u")
                with c5: pwd  = st.text_input("Password", type="password", key="my_pw")
                conn_uri = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}" if db and user else ""
        with db_col3:
            st.markdown("<div style='padding-top:1.55rem'>", unsafe_allow_html=True)
            st.markdown('<div class="analyse-wrap">', unsafe_allow_html=True)
            connect_clicked = st.button("🔌 Connect & Analyse",
                                         key="db_connect_btn",
                                         use_container_width=True,
                                         disabled=not bool(conn_uri))
            st.markdown("</div></div>", unsafe_allow_html=True)

        if connect_clicked and conn_uri:
            with st.spinner("Connecting to database…"):
                try:
                    if db_type == "SQLite":
                        all_ds = load_sqlite_path(conn_uri)
                    else:
                        all_ds = load_db_uri(conn_uri)
                    report  = build_analysis_report(all_ds)
                    dyn_rag = DynamicRAG(report)
                    st.session_state.datasets   = all_ds
                    st.session_state.report     = report
                    st.session_state.dyn_rag    = dyn_rag
                    st.session_state.mode       = "dynamic"
                    st.session_state.messages   = []
                    st.session_state.input_key += 1
                    s = report["summary"]
                    html = (
                        f"<strong>✅ DB connected — {len(all_ds)} table(s) loaded</strong>"
                        f"<div class='sc-row' style='margin-top:8px'>"
                        + "".join(
                            f"<div class='sc'><div class='sc-l'>{lb}</div>"
                            f"<div class='sc-v'>{vl}</div></div>"
                            for lb,vl in [("Tables",len(all_ds)),
                                          ("Rows",f"{s['total_rows']:,}"),
                                          ("Relations",s["total_relationships"]),
                                          ("DQ issues",s["total_dq_issues"])]
                        )
                        + "</div>"
                    )
                    btns = (
                        [("📊 Overview","Show dataset overview"),
                         ("🔗 Relationships","Show all relationships between datasets"),
                         ("🛡️ DQ Issues","Show all data quality issues")]
                        + [(f"📋 {d.name}", f"Show schema of {d.name}") for d in all_ds[:5]]
                    )
                    st.session_state.messages.append(
                        {"role":"ai","html":html,"btns":btns,"sources":[]}
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Connection failed: {e}")

    # ── TAB: CDISC Demo ───────────────────────────────────────────────────────
    else:
        demo_col1, demo_col2 = st.columns([3, 1])
        with demo_col1:
            st.markdown("""
            <div style="font-size:.82rem;color:#475569;line-height:1.7;padding:.2rem 0">
            Built-in demo: <strong>STUDY-2026-FIBER</strong> — Phase III RCT clinical trial.<br>
            5 CDISC tables (DM, AE, VS, LB, ADSL) · 10,240 records · 78 knowledge chunks.<br>
            Ask about schemas, DQ issues, ER relationships, glossary terms, and statistics.
            </div>
            """, unsafe_allow_html=True)
        with demo_col2:
            if st.session_state.mode != "static":
                st.markdown('<div class="analyse-wrap">', unsafe_allow_html=True)
                if st.button("🧬 Use CDISC Demo", key="use_demo_btn", use_container_width=True):
                    st.session_state.mode      = "static"
                    st.session_state.messages  = []
                    st.session_state.input_key += 1
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="background:#f0fdf4;border:1px solid #86efac;border-radius:8px;'
                    'padding:5px 10px;font-size:.75rem;color:#15803d;font-weight:600">'
                    '✅ CDISC demo active</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("</div>", unsafe_allow_html=True)  # close upload-panel


# ─────────────────────────── Quick chips ─────────────────────────────────────
if st.session_state.mode == "dynamic" and st.session_state.report:
    ds_list = st.session_state.datasets
    CHIPS = (
        [("📊 Overview",      "Show dataset overview"),
         ("🔗 Relationships", "Show all relationships between datasets"),
         ("🛡️ DQ Issues",    "Show all data quality issues")]
        + [(f"📋 {d.name}", f"Show schema of {d.name}") for d in ds_list[:5]]
    )
else:
    CHIPS = [
        ("🛡️ DQ Audit",    "Show all data quality issues and remediations"),
        ("📋 SDTM Tables", "List all SDTM and ADaM tables"),
        ("🔗 ER Diagram",  "Explain ER relationships between all tables"),
        ("📖 Dictionary",  "Show full data dictionary for all columns"),
        ("📚 Glossary",    "List all CDISC and GCP glossary terms"),
        ("📊 Dashboard",   "Show DQ completeness metrics and error trends"),
        ("📜 Reg Report",  "Show regulatory submission status and dossier"),
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


# ─────────────────────────── Welcome ─────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 1rem .8rem;color:#64748b">
      <div style="font-size:1.8rem;margin-bottom:.35rem">🧬</div>
      <div style="font-size:.95rem;font-weight:800;color:#0f172a;margin-bottom:.4rem">
        DataGenome AI — Upload your data and start chatting
      </div>
      <div style="font-size:.81rem;line-height:1.65;max-width:560px;margin:0 auto .8rem">
        Step 1 → Choose a tab above &nbsp;·&nbsp;
        Step 2 → Upload files or connect a DB &nbsp;·&nbsp;
        Step 3 → Click <strong style="color:#7c3aed">Analyse</strong> &nbsp;·&nbsp;
        Step 4 → Ask anything
      </div>
      <div style="display:flex;flex-wrap:wrap;justify-content:center;gap:4px">
        <span style="background:#f1f5f9;border:1px solid #e2e8f0;border-radius:8px;
          padding:4px 10px;font-size:.74rem;color:#475569">
          💡 "What is the max age in employees?"</span>
        <span style="background:#f1f5f9;border:1px solid #e2e8f0;border-radius:8px;
          padding:4px 10px;font-size:.74rem;color:#475569">
          💡 "Show schema of orders"</span>
        <span style="background:#f1f5f9;border:1px solid #e2e8f0;border-radius:8px;
          padding:4px 10px;font-size:.74rem;color:#475569">
          💡 "Are there relationships between tables?"</span>
        <span style="background:#f1f5f9;border:1px solid #e2e8f0;border-radius:8px;
          padding:4px 10px;font-size:.74rem;color:#475569">
          💡 "Show missing values"</span>
        <span style="background:#f1f5f9;border:1px solid #e2e8f0;border-radius:8px;
          padding:4px 10px;font-size:.74rem;color:#475569">
          💡 "Distribution of status column"</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────── Chat messages ───────────────────────────────────
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
                cat   = s.get("category", "info")
                score = s.get("score", "—")
                title = s.get("title", "")
                short = title[:42] + ("…" if len(title) > 42 else "")
                return (f'<span class="src-chip {cat}" title="score:{score}">'
                        f'{short}</span>')
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


# ─────────────────────────── Input ───────────────────────────────────────────
st.markdown("<hr style='margin:.2rem 0 0;border-color:#e9eef4'>", unsafe_allow_html=True)
in_col, btn_col = st.columns([6, 1])
with in_col:
    ph = ("Ask about your data… e.g. 'max salary', 'show schema of orders', "
          "'missing values in customers'"
          if st.session_state.mode == "dynamic"
          else "Ask about CDISC schemas, DQ issues, statistics, glossary…")
    user_input = st.text_area(
        "Message",
        label_visibility="collapsed",
        placeholder=ph,
        height=66,
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
    "<div style='text-align:center;font-size:.64rem;color:#cbd5e1;padding:.2rem 0 .4rem'>"
    "BM25 retrieval · pandas profiling · No API key · "
    "Free · GitHub + Streamlit Cloud ready"
    "</div>",
    unsafe_allow_html=True,
)
