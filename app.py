"""
DataGenome AI — Decoding the DNA of Life Sciences Data
Complete implementation of all 8 abstract capabilities via conversational chatbot.
No SQL knowledge required. No feature buttons — everything through natural language.
"""
import io, base64, re
import streamlit as st
import pandas as pd

from core import load_file, build_report, build_rag, profile_df, LS_GLOSSARY
from responder import respond, detect_intent

st.set_page_config(
    page_title="DataGenome AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS — complete design system
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
*,*::before,*::after{box-sizing:border-box}
#MainMenu,footer,header{visibility:hidden}
[data-testid="collapsedControl"]{display:none}
.block-container{padding:0!important;max-width:100%!important}

/* ── Top bar ── */
.topbar{
  background:linear-gradient(135deg,#0f172a 0%,#0d4a45 60%,#134e4a 100%);
  color:white;padding:.55rem 1.4rem;
  display:flex;align-items:center;gap:.75rem;flex-wrap:wrap
}
.topbar-logo{font-size:1.35rem}
.topbar-title{font-size:1.02rem;font-weight:800;letter-spacing:-.3px}
.topbar-tag{font-size:.68rem;opacity:.5;font-style:italic}
.topbar-right{margin-left:auto;font-size:.68rem;opacity:.6;
  display:flex;align-items:center;gap:6px}
.dot{width:7px;height:7px;border-radius:50%;background:#22c55e;display:inline-block}

/* ── Upload panel ── */
.upload-panel{
  background:#fff;border-bottom:1px solid #e2e8f0;
  padding:.65rem 1.4rem .5rem
}
.up-title{font-size:.8rem;font-weight:700;color:#0f172a;
  display:flex;align-items:center;gap:6px;margin-bottom:.4rem}

/* ── Capability badges ── */
.cap-bar{
  display:flex;flex-wrap:wrap;gap:4px;
  padding:.35rem 1.4rem;background:#f8fafc;
  border-bottom:1px solid #e9eef4
}
.cap-badge{
  display:inline-flex;align-items:center;gap:3px;
  background:#fff;border:1px solid #e2e8f0;border-radius:20px;
  padding:2px 10px;font-size:.67rem;color:#475569;font-weight:600
}
.cap-badge.active{background:#0d9488;color:white;border-color:#0d9488}

/* ── Bubbles ── */
.bubble-user{
  background:#0d9488;color:#fff!important;
  padding:.6rem 1rem;border-radius:14px 14px 3px 14px;
  margin:.4rem 0 .4rem auto;max-width:76%;
  width:fit-content;font-size:.87rem;line-height:1.55;
  display:block;word-break:break-word
}
.bubble-ai{
  background:#fff;color:#1e293b!important;
  border:1px solid #e2e8f0;border-left:3px solid #0d9488;
  padding:.8rem 1rem;border-radius:3px 14px 14px 14px;
  margin:.4rem 0;width:100%;font-size:.85rem;
  line-height:1.7;display:block;word-break:break-word
}
.bubble-ai strong{color:#0f172a}
.bubble-ai code{
  background:#f1f5f9;color:#0f766e!important;
  padding:1px 5px;border-radius:4px;font-size:.77rem
}
.bubble-ai pre{
  background:#f1f5f9;padding:6px 10px;border-radius:6px;
  font-size:.72rem;overflow-x:auto;color:#334155;margin-top:5px
}
.bubble-ai table{width:100%;border-collapse:collapse;margin-top:6px}
.bubble-ai th{
  background:#f8fafc;padding:5px 8px;text-align:left;
  font-size:.69rem;color:#64748b;border-bottom:1px solid #e2e8f0;
  white-space:nowrap
}
.bubble-ai td{
  padding:4px 8px;font-size:.74rem;
  border-bottom:1px solid #f1f5f9;vertical-align:top
}

/* ── Stat cards ── */
.sc-row{display:flex;gap:5px;flex-wrap:wrap;margin:6px 0}
.sc{
  background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
  padding:5px 10px;min-width:70px;flex:1 1 auto
}
.sc-l{font-size:.61rem;color:#64748b;margin-bottom:1px}
.sc-v{font-size:.88rem;font-weight:700;color:#0f172a}

/* ── Bar ── */
.bar-row{display:flex;align-items:center;gap:6px;margin:3px 0}
.bar-bg{flex:1;background:#f1f5f9;border-radius:4px;height:11px;overflow:hidden}
.bar-fill{height:100%;border-radius:4px}

/* ── Source strip ── */
.src-strip{
  background:#f8fafc;border:1px solid #e9eef4;border-top:none;
  border-radius:0 0 8px 8px;padding:4px 10px;
  font-size:.66rem;color:#94a3b8;
  display:flex;flex-wrap:wrap;gap:3px;align-items:center
}
.src-chip{
  border-radius:10px;padding:1px 7px;font-size:.64rem;font-weight:600;
  background:#e0f2fe;color:#0369a1
}
.src-chip.dq{background:#fee2e2;color:#dc2626}
.src-chip.schema,.src-chip.column{background:#ccfbf1;color:#0f766e}
.src-chip.relationship{background:#dbeafe;color:#1d4ed8}
.src-chip.overview{background:#f3e8ff;color:#7c3aed}
.src-chip.glossary{background:#fef9c3;color:#854d0e}

/* ── Download action card ── */
.dl-card{
  background:#f0fdf4;border:1px solid #86efac;border-radius:10px;
  padding:.6rem .9rem;margin-top:6px;
  display:flex;align-items:center;gap:10px;font-size:.8rem;color:#15803d
}
.dl-label{font-weight:700;flex:1}

/* ── Buttons ── */
div[data-testid="stButton"]>button{
  width:100%!important;text-align:left!important;
  justify-content:flex-start!important;
  background:#f8fafc!important;border:1px solid #e2e8f0!important;
  border-radius:8px!important;color:#334155!important;
  font-size:.78rem!important;padding:.35rem .7rem!important;
  white-space:normal!important;word-break:break-word!important;
  height:auto!important;min-height:30px!important;line-height:1.4!important
}
div[data-testid="stButton"]>button:hover{
  border-color:#0d9488!important;background:#f0fdf4!important;color:#0f172a!important
}

/* Send button */
.send-wrap div[data-testid="stButton"]>button{
  background:#0d9488!important;color:#fff!important;border:none!important;
  font-weight:700!important;border-radius:10px!important;
  justify-content:center!important;font-size:.88rem!important
}
.send-wrap div[data-testid="stButton"]>button:hover{background:#0f766e!important}

/* Analyse button */
.analyse-wrap div[data-testid="stButton"]>button{
  background:#7c3aed!important;color:#fff!important;border:none!important;
  font-weight:700!important;border-radius:10px!important;
  justify-content:center!important
}
.analyse-wrap div[data-testid="stButton"]>button:hover{background:#6d28d9!important}

/* Clear button */
.clear-wrap div[data-testid="stButton"]>button{
  background:#fff!important;color:#dc2626!important;
  border:1px solid #fca5a5!important;font-size:.75rem!important;
  border-radius:8px!important;justify-content:center!important
}

/* Textarea */
.stTextArea>div>div>textarea{
  border-radius:10px!important;border:1.5px solid #cbd5e1!important;
  font-size:.87rem!important;resize:none!important;
  background:#fff!important;color:#0f172a!important;
  caret-color:#0d9488!important;padding:9px 12px!important;line-height:1.5!important
}
.stTextArea>div>div>textarea:focus{
  border-color:#0d9488!important;outline:none!important;
  box-shadow:0 0 0 2px rgba(13,148,136,.15)!important;color:#0f172a!important
}
.stTextArea>div>div>textarea::placeholder{color:#94a3b8!important;opacity:1!important}

/* Text inputs */
.stTextInput>div>div>input{
  border-radius:8px!important;border:1.5px solid #cbd5e1!important;
  background:#fff!important;color:#0f172a!important;
  font-size:.84rem!important;padding:5px 10px!important
}
.stTextInput>div>div>input:focus{border-color:#0d9488!important;color:#0f172a!important}

/* Selectbox */
.stSelectbox>div>div{
  border-radius:8px!important;border:1.5px solid #cbd5e1!important;
  background:#fff!important;color:#0f172a!important
}

/* File uploader */
[data-testid="stFileUploader"]{
  border:2px dashed #cbd5e1;border-radius:10px;
  padding:.4rem;background:#fafafa
}
[data-testid="stFileUploader"]:hover{border-color:#0d9488;background:#f0fdf4}

/* Radio */
.stRadio>div{gap:6px!important}
.stRadio>div>label{
  background:#f8fafc!important;border:1px solid #e2e8f0!important;
  border-radius:8px!important;padding:4px 12px!important;
  font-size:.78rem!important;cursor:pointer
}

/* Status */
.status-ok{
  background:#f0fdf4;border:1px solid #86efac;border-radius:8px;
  padding:4px 10px;font-size:.73rem;color:#15803d;font-weight:600
}
.status-pending{
  background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;
  padding:4px 10px;font-size:.73rem;color:#c2410c
}

/* Welcome tips */
.tip-row{display:flex;flex-wrap:wrap;gap:5px;justify-content:center;margin-top:.6rem}
.tip{
  background:#f1f5f9;border:1px solid #e2e8f0;border-radius:8px;
  padding:4px 10px;font-size:.73rem;color:#475569;cursor:default
}

::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:#f8fafc}
::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:3px}

@media(max-width:680px){
  .bubble-user,.bubble-ai{max-width:98%;font-size:.82rem}
  .topbar-right{display:none}
  .sc{min-width:60px}
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════════════════
def _init():
    for k, v in {
        "messages":  [],
        "pending":   "",
        "input_key": 0,
        "tables":    [],       # list[TableProfile]
        "report":    None,
        "rag_idx":   None,
        "rag_chunks":[],
        "mode":      "demo",   # "demo" | "user"
        "src_tab":   "upload", # "upload" | "db" | "demo"
        "dl_queue":  [],       # [(label, bytes, mime, filename)]
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()

# ── Load demo data on first run ───────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_demo():
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    tables = []
    for fname in ["patients.csv", "lab_results.csv"]:
        path = os.path.join(base, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            tname = fname.replace(".csv","")
            tables.append(profile_df(df, tname, "demo-csv"))
    rep = build_report(tables)
    idx, chunks = build_rag(rep)
    return tables, rep, idx, chunks

def _ensure_demo():
    if st.session_state.mode == "demo" and not st.session_state.report:
        tables, rep, idx, chunks = _load_demo()
        st.session_state.tables    = tables
        st.session_state.report    = rep
        st.session_state.rag_idx   = idx
        st.session_state.rag_chunks= chunks
_ensure_demo()

# ══════════════════════════════════════════════════════════════════════════════
# Submit logic
# ══════════════════════════════════════════════════════════════════════════════
def _submit(q: str):
    q = q.strip()
    if not q or not st.session_state.report:
        return

    result = respond(q, st.session_state.report,
                     st.session_state.rag_idx,
                     st.session_state.rag_chunks)

    st.session_state.messages.append({"role":"user","text":q})
    ai_msg = {
        "role":   "ai",
        "html":   result["html"],
        "action": result.get("action"),
        "chunks": result.get("chunks",[]),
    }

    # Queue download if action
    if result.get("action") == "pdf" and result.get("action_data"):
        st.session_state.dl_queue.append((
            "📄 Download PDF Report",
            result["action_data"],
            "application/pdf",
            "datagenome_report.pdf",
        ))
    elif result.get("action") == "dict_csv" and result.get("action_data"):
        st.session_state.dl_queue.append((
            "📥 Download Data Dictionary CSV",
            result["action_data"],
            "text/csv",
            "data_dictionary.csv",
        ))

    st.session_state.messages.append(ai_msg)
    st.session_state.pending   = ""
    st.session_state.input_key += 1

if st.session_state.pending:
    _submit(st.session_state.pending)

# ══════════════════════════════════════════════════════════════════════════════
# TOP BAR
# ══════════════════════════════════════════════════════════════════════════════
s    = st.session_state.report["summary"] if st.session_state.report else {}
mode = "Demo data" if st.session_state.mode=="demo" else "Your data"
n_chunks = len(st.session_state.rag_chunks)

st.markdown(f"""
<div class="topbar">
  <span class="topbar-logo">🧬</span>
  <div>
    <div class="topbar-title">DataGenome AI</div>
    <div class="topbar-tag">Decoding the DNA of Life Sciences Data</div>
  </div>
  <div class="topbar-right">
    <span class="dot"></span>
    {mode} &nbsp;·&nbsp;
    {s.get('n_tables',0)} tables &nbsp;·&nbsp;
    {s.get('total_rows',0):,} rows &nbsp;·&nbsp;
    {n_chunks} RAG chunks
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA SOURCE PANEL
# ══════════════════════════════════════════════════════════════════════════════
with st.container():
    st.markdown('<div class="upload-panel">', unsafe_allow_html=True)
    st.markdown('<div class="up-title">📂 Data Source — Upload files, connect a database, or use the built-in demo</div>', unsafe_allow_html=True)

    tab_c = st.columns([1,1,1,2])
    with tab_c[0]:
        if st.button("📁 Upload Files",
                     type="primary" if st.session_state.src_tab=="upload" else "secondary",
                     key="tab_up"):
            st.session_state.src_tab="upload"; st.rerun()
    with tab_c[1]:
        if st.button("🔌 Connect DB",
                     type="primary" if st.session_state.src_tab=="db" else "secondary",
                     key="tab_db"):
            st.session_state.src_tab="db"; st.rerun()
    with tab_c[2]:
        if st.button("🧬 Demo Data",
                     type="primary" if st.session_state.src_tab=="demo" else "secondary",
                     key="tab_demo"):
            st.session_state.src_tab="demo"; st.rerun()
    with tab_c[3]:
        # Status indicator
        if st.session_state.report:
            tnames = " · ".join(f"`{t.name}`({t.row_count:,}r)" for t in st.session_state.tables)
            st.markdown(f'<div class="status-ok">✅ Ready — {tnames}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-pending">⏳ No data loaded</div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top:.45rem'>", unsafe_allow_html=True)

    # ── Upload tab ────────────────────────────────────────────────────────────
    if st.session_state.src_tab == "upload":
        up_c, btn_c = st.columns([5,1])
        with up_c:
            uploaded = st.file_uploader(
                "CSV · TSV · Excel · JSON · SQLite — drop multiple files for cross-table analysis",
                type=["csv","tsv","txt","xlsx","xls","json","db","sqlite","sqlite3"],
                accept_multiple_files=True,
                label_visibility="visible",
                key="uploader",
            )
        with btn_c:
            if uploaded:
                st.markdown('<div class="analyse-wrap" style="padding-top:1.55rem">', unsafe_allow_html=True)
                if st.button("🔍 Analyse", key="analyse_btn", use_container_width=True):
                    with st.spinner("Profiling datasets…"):
                        all_tables, errs = [], []
                        for f in uploaded:
                            try:
                                f.seek(0)
                                all_tables.extend(load_file(f))
                            except Exception as e:
                                errs.append(f"{f.name}: {e}")
                    for e in errs:
                        st.error(e)
                    if all_tables:
                        with st.spinner("Building RAG index & detecting relationships…"):
                            rep = build_report(all_tables)
                            idx, chunks = build_rag(rep)
                        st.session_state.tables     = all_tables
                        st.session_state.report     = rep
                        st.session_state.rag_idx    = idx
                        st.session_state.rag_chunks = chunks
                        st.session_state.mode       = "user"
                        st.session_state.messages   = []
                        st.session_state.dl_queue   = []
                        st.session_state.input_key += 1
                        ss = rep["summary"]
                        # Welcome message
                        tns = ", ".join(f"`{t.name}`" for t in all_tables)
                        welcome_html = (
                            f"<strong>✅ {len(all_tables)} dataset(s) analysed — all 8 capabilities ready!</strong><br>"
                            f"<span style='font-size:.78rem;color:#64748b'>Tables: {tns}</span>"
                            f"<div class='sc-row' style='margin-top:8px'>"
                            + "".join(
                                f"<div class='sc'><div class='sc-l'>{lb}</div>"
                                f"<div class='sc-v' style='color:{cl}'>{vl}</div></div>"
                                for lb,vl,cl in [
                                    ("Datasets",     len(all_tables),             "#0f172a"),
                                    ("Total rows",   f"{ss['total_rows']:,}",     "#0f172a"),
                                    ("Total cols",   ss["total_cols"],             "#0f172a"),
                                    ("Relationships",ss["n_rels"],                "#0d9488"),
                                    ("DQ issues",    ss["n_dq"],
                                     "#dc2626" if ss["n_crit"]>0 else "#ca8a04"),
                                ]
                            )
                            + "</div>"
                            + "<div style='margin-top:8px;font-size:.78rem;color:#475569'>"
                            + "Try: <em>'create dashboard'</em> · <em>'show ER diagram'</em> · "
                            + "<em>'generate PDF report'</em> · <em>'explain patients table'</em> · "
                            + "<em>'what is the max age?'</em> · <em>'export data dictionary'</em>"
                            + "</div>"
                        )
                        st.session_state.messages.append({
                            "role":"ai","html":welcome_html,
                            "action":None,"chunks":[]
                        })
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

    # ── DB tab ────────────────────────────────────────────────────────────────
    elif st.session_state.src_tab == "db":
        db_c1, db_c2, db_c3 = st.columns([1,3,1])
        with db_c1:
            db_type = st.selectbox("Type",["SQLite","PostgreSQL","MySQL"],key="db_type")
        with db_c2:
            if db_type == "SQLite":
                db_path = st.text_input("File path",placeholder="/path/to/database.db",key="sqlite_path")
                conn_uri = db_path.strip() if db_path else ""
            else:
                c1,c2,c3,c4,c5 = st.columns([2,1,2,1,1])
                host = c1.text_input("Host","localhost",key="db_host")
                port = c2.text_input("Port","5432" if db_type=="PostgreSQL" else "3306",key="db_port")
                db   = c3.text_input("Database",key="db_name")
                user = c4.text_input("User",key="db_user")
                pwd  = c5.text_input("Password",type="password",key="db_pwd")
                prefix = "postgresql" if db_type=="PostgreSQL" else "mysql+pymysql"
                conn_uri = f"{prefix}://{user}:{pwd}@{host}:{port}/{db}" if db and user else ""
        with db_c3:
            st.markdown("<div style='padding-top:1.55rem'>", unsafe_allow_html=True)
            st.markdown('<div class="analyse-wrap">', unsafe_allow_html=True)
            if st.button("🔌 Connect",key="db_connect",use_container_width=True,disabled=not bool(conn_uri)):
                with st.spinner("Connecting…"):
                    try:
                        import sqlite3
                        if db_type=="SQLite":
                            import tempfile, os
                            conn = sqlite3.connect(conn_uri)
                            tbls = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'",conn)
                            all_tables = []
                            for tn in tbls["name"].tolist():
                                df = pd.read_sql(f'SELECT * FROM "{tn}"',conn)
                                all_tables.append(profile_df(df,tn,"sqlite"))
                            conn.close()
                        else:
                            from sqlalchemy import create_engine, inspect
                            eng  = create_engine(conn_uri,connect_args={"connect_timeout":8})
                            insp = inspect(eng)
                            all_tables = []
                            with eng.connect() as c:
                                for tn in insp.get_table_names():
                                    df = pd.read_sql_table(tn,c)
                                    all_tables.append(profile_df(df,tn,db_type.lower()))
                        rep = build_report(all_tables)
                        idx,chunks = build_rag(rep)
                        st.session_state.tables=all_tables; st.session_state.report=rep
                        st.session_state.rag_idx=idx; st.session_state.rag_chunks=chunks
                        st.session_state.mode="user"; st.session_state.messages=[]
                        st.session_state.dl_queue=[]; st.session_state.input_key+=1
                        st.rerun()
                    except Exception as e:
                        st.error(f"Connection failed: {e}")
            st.markdown("</div></div>", unsafe_allow_html=True)

    # ── Demo tab ──────────────────────────────────────────────────────────────
    else:
        d1,d2 = st.columns([4,1])
        with d1:
            st.markdown("""
            <div style='font-size:.81rem;color:#475569;line-height:1.7;padding:.1rem 0'>
            Built-in life sciences demo: <strong>patients</strong> (30 subjects, 18 clinical variables)
            + <strong>lab_results</strong> (50 records, 12 lab parameters).<br>
            Phase III RCT prototype — DRUG_X vs PLACEBO · ALT · AST · WBC · HGB · Creatinine.
            </div>
            """, unsafe_allow_html=True)
        with d2:
            if st.session_state.mode != "demo":
                st.markdown('<div class="analyse-wrap">', unsafe_allow_html=True)
                if st.button("Use Demo",key="use_demo",use_container_width=True):
                    tables,rep,idx,chunks = _load_demo()
                    st.session_state.tables=tables; st.session_state.report=rep
                    st.session_state.rag_idx=idx; st.session_state.rag_chunks=chunks
                    st.session_state.mode="demo"; st.session_state.messages=[]
                    st.session_state.dl_queue=[]; st.session_state.input_key+=1
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-ok">✅ Active</div>', unsafe_allow_html=True)

    if st.session_state.report:
        cl_c1, cl_c2 = st.columns([6,1])
        with cl_c2:
            st.markdown('<div class="clear-wrap">', unsafe_allow_html=True)
            if st.button("🗑️ Clear",key="clear_btn"):
                for k in ["tables","report","rag_idx","rag_chunks","messages","dl_queue"]:
                    st.session_state[k] = [] if isinstance(st.session_state[k],list) else None
                st.session_state.mode="demo"; st.session_state.input_key+=1
                _ensure_demo(); st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CAPABILITY BAR  (shows which abstract capabilities are fulfilled)
# ══════════════════════════════════════════════════════════════════════════════
CAPS = [
    ("🔍","Schema Intelligence"),
    ("📚","Business Glossary"),
    ("🛡️","DQ Audit"),
    ("🔗","ER Diagram"),
    ("📊","Analytics Dashboard"),
    ("📖","Data Dictionary"),
    ("📄","Regulatory PDF"),
    ("💬","Data Q&A"),
]
badge_html = "".join(
    f'<span class="cap-badge {"active" if st.session_state.report else ""}">{icon} {lbl}</span>'
    for icon,lbl in CAPS
)
st.markdown(f'<div class="cap-bar">{badge_html}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# WELCOME SCREEN
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.messages:
    name_str = " + ".join(t.name for t in st.session_state.tables) if st.session_state.tables else "demo"
    st.markdown(f"""
    <div style="text-align:center;padding:1.8rem 1rem .8rem;color:#64748b">
      <div style="font-size:2.2rem;margin-bottom:.3rem">🧬</div>
      <div style="font-size:.98rem;font-weight:800;color:#0f172a;margin-bottom:.4rem">
        DataGenome AI — All 8 Capabilities Active
      </div>
      <div style="font-size:.8rem;line-height:1.65;max-width:620px;margin:0 auto .7rem;color:#475569">
        Loaded: <strong>{name_str}</strong>. Ask anything in plain English —
        schema exploration, DQ audit, ER diagram, analytics dashboard,
        data dictionary, PDF report, business glossary, or statistical Q&amp;A.
      </div>
      <div class="tip-row">
        <span class="tip">💡 explain the patients table</span>
        <span class="tip">💡 create a dashboard</span>
        <span class="tip">💡 show ER diagram</span>
        <span class="tip">💡 generate PDF report</span>
        <span class="tip">💡 what is the max age?</span>
        <span class="tip">💡 show DQ issues</span>
        <span class="tip">💡 export data dictionary</span>
        <span class="tip">💡 what is ALT?</span>
        <span class="tip">💡 relationship between tables</span>
        <span class="tip">💡 distribution of treatment arm</span>
        <span class="tip">💡 show missing values in lab_results</span>
        <span class="tip">💡 dashboard for age vs response score</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CHAT MESSAGES
# ══════════════════════════════════════════════════════════════════════════════
chat_wrap = st.container()
with chat_wrap:
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
            # Source strip
            chunks = msg.get("chunks",[])
            if chunks:
                def _chip(c):
                    cat   = c.get("cat","info")
                    score = c.get("score","—")
                    title = c.get("title","")
                    short = title[:40]+("…" if len(title)>40 else "")
                    return f'<span class="src-chip {cat}" title="BM25 score:{score}">{short}</span>'
                chips = "".join(_chip(c) for c in chunks[:5])
                st.markdown(f'<div class="src-strip">📎 RAG sources: {chips}</div>',
                            unsafe_allow_html=True)

            # Download button (PDF or CSV)
            if msg.get("action") in ("pdf","dict_csv") and st.session_state.dl_queue:
                # Find the matching download
                for dl in st.session_state.dl_queue[-3:]:
                    lbl, data, mime, fname = dl
                    st.download_button(
                        label=lbl,
                        data=data,
                        file_name=fname,
                        mime=mime,
                        key=f"dl_{idx}_{fname}",
                    )

# ══════════════════════════════════════════════════════════════════════════════
# INPUT ROW
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr style='margin:.2rem 0 0;border-color:#e9eef4'>", unsafe_allow_html=True)

in_c, btn_c = st.columns([6,1])
with in_c:
    ph = ("Ask anything… e.g. 'create dashboard', 'show ER diagram', "
          "'generate PDF', 'what is max age?', 'explain lab_results table'")
    user_input = st.text_area(
        "Message",
        label_visibility="collapsed",
        placeholder=ph,
        height=66,
        key=f"inp_{st.session_state.input_key}",
    )
with btn_c:
    st.markdown("<div class='send-wrap' style='padding-top:4px'>", unsafe_allow_html=True)
    send = st.button("Send ▶", key="send_btn", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

if send and user_input and user_input.strip():
    if not st.session_state.report:
        st.warning("Please load data first — upload a file or use the Demo tab.")
    else:
        _submit(user_input)
        st.rerun()

st.markdown(
    "<div style='text-align:center;font-size:.63rem;color:#cbd5e1;padding:.18rem 0 .38rem'>"
    "DataGenome AI · Cognizant BlueBolt · GenAI for Life Sciences · "
    "BM25 RAG · Plotly · FPDF2 · No API key · Free · Streamlit Cloud ready"
    "</div>",
    unsafe_allow_html=True,
)
