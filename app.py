"""
DataGenome AI — Clean RAG Chatbot
Everything triggered by user query. No default panels. Upload → Analyse → Ask.
"""
import re, io, os
import streamlit as st
import pandas as pd
import plotly.express as px

from data_loader import load_uploaded_file, load_db_uri, load_sqlite_path, load_csv
from analyzer import build_analysis_report
from dynamic_rag import DynamicRAG
from dynamic_answer import generate_dynamic_answer
from chart_engine import resolve_chart_request, is_chart_query, build_auto_key_dashboard
from stats_engine import answer_stats_query
from rag_engine import ClinicalRAG
from answer_engine import generate_answer
from data import CLINICAL_TABLES, MOCK_DQ_ISSUES, GLOSSARY, DATASET_STATS

st.set_page_config(page_title="DataGenome AI", page_icon="🧬",
                   layout="wide", initial_sidebar_state="collapsed")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""<style>
*,*::before,*::after{box-sizing:border-box}
#MainMenu,footer,header,[data-testid="collapsedControl"]{visibility:hidden;display:none}
.block-container{padding:0!important;max-width:100%!important}

.topbar{background:linear-gradient(135deg,#0f172a,#134e4a);color:#fff;
  padding:.5rem 1.2rem;display:flex;align-items:center;gap:.6rem;flex-wrap:wrap}
.t-title{font-size:1rem;font-weight:800}
.t-sub{font-size:.68rem;opacity:.4}
.t-right{margin-left:auto;font-size:.68rem;opacity:.55;display:flex;align-items:center;gap:5px}
.dot{width:7px;height:7px;border-radius:50%;background:#22c55e;display:inline-block}

.bubble-user{background:#0d9488;color:#fff!important;padding:.58rem .95rem;
  border-radius:14px 14px 3px 14px;margin:.35rem 0 .35rem auto;
  max-width:76%;width:fit-content;font-size:.87rem;line-height:1.55;
  display:block;word-break:break-word}
.bubble-ai{background:#fff;color:#1e293b!important;border:1px solid #e2e8f0;
  border-left:3px solid #0d9488;padding:.75rem 1rem;
  border-radius:3px 14px 14px 14px;margin:.35rem 0;
  max-width:100%;font-size:.86rem;line-height:1.7;
  display:block;word-break:break-word}
.bubble-ai strong{color:#0f172a}
.bubble-ai code{background:#f1f5f9;color:#0f766e!important;
  padding:1px 5px;border-radius:4px;font-size:.78rem}

.sc-row{display:flex;gap:5px;flex-wrap:wrap;margin:6px 0}
.sc{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
  padding:5px 10px;min-width:70px;flex:1 1 auto}
.sc-l{font-size:.62rem;color:#64748b;margin-bottom:1px}
.sc-v{font-size:.9rem;font-weight:700;color:#0f172a}

.bar-row{display:flex;align-items:center;gap:6px;margin:3px 0}
.bar-bg{flex:1;background:#f1f5f9;border-radius:4px;height:11px;overflow:hidden}
.bar-fill{height:100%;border-radius:4px}

.sources{background:#f8fafc;border:1px solid #e9eef4;border-top:none;
  border-radius:0 0 8px 8px;padding:4px 10px;font-size:.67rem;color:#94a3b8;
  display:flex;flex-wrap:wrap;gap:3px;align-items:center}
.src-chip{background:#e0f2fe;color:#0369a1;border-radius:10px;
  padding:1px 7px;font-size:.64rem;font-weight:600}
.src-chip.dq{background:#fee2e2;color:#dc2626}
.src-chip.schema,.src-chip.column{background:#ccfbf1;color:#0f766e}
.src-chip.relationship{background:#dbeafe;color:#1d4ed8}
.src-chip.overview{background:#f3e8ff;color:#7c3aed}
.src-chip.stats,.src-chip.glossary{background:#fef9c3;color:#854d0e}
.src-chip.chart{background:#fce7f3;color:#be185d}

div[data-testid="stButton"]>button{
  width:100%!important;text-align:left!important;justify-content:flex-start!important;
  background:#f8fafc!important;border:1px solid #e2e8f0!important;border-radius:8px!important;
  color:#334155!important;font-size:.79rem!important;padding:.38rem .72rem!important;
  white-space:normal!important;word-break:break-word!important;
  height:auto!important;min-height:32px!important;line-height:1.4!important}
div[data-testid="stButton"]>button:hover{
  border-color:#0d9488!important;background:#f0fdf4!important;color:#0f172a!important}

.send-wrap div[data-testid="stButton"]>button{
  background:#0d9488!important;color:#fff!important;border:none!important;
  font-weight:700!important;border-radius:10px!important;
  justify-content:center!important;font-size:.88rem!important}
.send-wrap div[data-testid="stButton"]>button:hover{background:#0f766e!important}

.analyse-wrap div[data-testid="stButton"]>button{
  background:#7c3aed!important;color:#fff!important;border:none!important;
  font-weight:700!important;border-radius:10px!important;
  justify-content:center!important;font-size:.84rem!important}

.clear-wrap div[data-testid="stButton"]>button{
  background:#fff!important;color:#dc2626!important;
  border:1px solid #fca5a5!important;font-weight:600!important;
  border-radius:8px!important;justify-content:center!important;font-size:.76rem!important}

.stTextArea>div>div>textarea{
  border-radius:10px!important;border:1.5px solid #cbd5e1!important;
  font-size:.88rem!important;resize:none!important;background:#fff!important;
  color:#0f172a!important;caret-color:#0d9488!important;
  padding:10px 12px!important;line-height:1.5!important}
.stTextArea>div>div>textarea:focus{
  border-color:#0d9488!important;outline:none!important;
  box-shadow:0 0 0 2px rgba(13,148,136,.15)!important;color:#0f172a!important}
.stTextArea>div>div>textarea::placeholder{color:#94a3b8!important;opacity:1!important}
.stTextInput>div>div>input{border-radius:8px!important;
  border:1.5px solid #cbd5e1!important;font-size:.85rem!important;
  background:#fff!important;color:#0f172a!important;padding:6px 10px!important}
.stTextInput>div>div>input:focus{border-color:#0d9488!important;color:#0f172a!important}

.upload-bar{background:#fff;border-bottom:1px solid #e2e8f0;
  padding:.6rem 1.2rem;display:flex;align-items:center;gap:.8rem;flex-wrap:wrap}
.status-pill{display:inline-flex;align-items:center;gap:4px;
  background:#f0fdf4;border:1px solid #86efac;border-radius:20px;
  padding:2px 9px;font-size:.72rem;color:#15803d;font-weight:600}
.status-pill-warn{background:#fff7ed;border-color:#fed7aa;color:#c2410c}

[data-testid="stFileUploader"]{border-radius:8px!important}
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:3px}
</style>""", unsafe_allow_html=True)

# ── Cached resources ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _static_rag():
    return ClinicalRAG()

@st.cache_data(show_spinner=False)
def _load_cdisc_datasets():
    """Convert built-in CLINICAL_TABLES data into DatasetInfo objects."""
    from data_loader import _profile_df
    datasets = []
    for t in CLINICAL_TABLES:
        rows = []
        n = t["rowCount"]
        import random, math
        random.seed(42)
        col_names = [c["name"] for c in t["columns"]]
        for i in range(min(n, 50)):                        # sample rows
            row = {}
            for c in t["columns"]:
                s = c["sampleData"]
                row[c["name"]] = s[i % len(s)]
            rows.append(row)
        df = pd.DataFrame(rows)
        ds = _profile_df(df, t["name"], "cdisc_demo")
        ds.description = t["description"]
        datasets.append(ds)
    return datasets

static_rag = _static_rag()

# ── Session init ───────────────────────────────────────────────────────────────
def _init():
    defs = {"messages":[],"pending":"","input_key":0,
            "datasets":[],"report":None,"dyn_rag":None,"mode":"welcome",
            "show_upload":True}
    for k,v in defs.items():
        if k not in st.session_state: st.session_state[k]=v
_init()

# ── Submit ─────────────────────────────────────────────────────────────────────
def submit(q:str):
    q=q.strip()
    if not q: return

    sources=[]
    # Chart / dashboard request
    if is_chart_query(q) and (st.session_state.mode=="dynamic" or st.session_state.mode=="cdisc"):
        datasets = st.session_state.datasets
        bundle   = resolve_chart_request(q, datasets)
        html = (f"<strong>📊 {bundle['title']}</strong><br>"
                f"<span style='font-size:.78rem;color:#64748b'>{bundle['summary']}</span>")
        btns = []
        sources = [{"category":"chart","title":f"Chart engine · {bundle['ds_name']}","score":1.0}]
        st.session_state.messages.append({"role":"user","text":q})
        st.session_state.messages.append({
            "role":"ai","html":html,"btns":btns,
            "sources":sources,"charts":bundle["charts"]})
        st.session_state.pending=""
        st.session_state.input_key+=1
        return

    if st.session_state.mode=="dynamic" and st.session_state.report:
        dyn    = st.session_state.dyn_rag
        chunks = dyn.retrieve(q, top_k=6)
        html, btns = generate_dynamic_answer(q, chunks, st.session_state.report)
        sources = chunks
    elif st.session_state.mode=="cdisc":
        stat = answer_stats_query(q)
        if stat:
            html, btns = stat
            sources = [{"category":"stats","title":"Stats Engine","score":1.0}]
        else:
            chunks = static_rag.retrieve(q, top_k=6)
            html, btns = generate_answer(q, chunks)
            sources = chunks
    else:
        # Welcome mode — just use static rag
        chunks = static_rag.retrieve(q, top_k=5)
        html, btns = generate_answer(q, chunks)
        sources = chunks

    st.session_state.messages.append({"role":"user","text":q})
    st.session_state.messages.append({"role":"ai","html":html,"btns":btns,
                                       "sources":sources,"charts":[]})
    st.session_state.pending=""
    st.session_state.input_key+=1

if st.session_state.pending:
    submit(st.session_state.pending)

# ── Helpers ────────────────────────────────────────────────────────────────────
def _run_analysis(all_ds):
    report  = build_analysis_report(all_ds)
    dyn_rag = DynamicRAG(report)
    st.session_state.datasets   = all_ds
    st.session_state.report     = report
    st.session_state.dyn_rag    = dyn_rag
    st.session_state.mode       = "dynamic"
    st.session_state.messages   = []
    st.session_state.input_key += 1
    s = report["summary"]
    tnames = " · ".join(f"<code>{d.name}</code>" for d in all_ds)
    html = (
        f"<strong>✅ {len(all_ds)} dataset(s) analysed and ready</strong><br>"
        f"<span style='font-size:.78rem;color:#64748b'>{tnames}</span>"
        f"<div class='sc-row' style='margin-top:8px'>"
        + "".join(f"<div class='sc'><div class='sc-l'>{lb}</div>"
                  f"<div class='sc-v' style='color:{cl}'>{vl}</div></div>"
                  for lb,vl,cl in [
                      ("Datasets",     len(all_ds),            "#0f172a"),
                      ("Total rows",   f"{s['total_rows']:,}", "#0f172a"),
                      ("Columns",      s["total_cols"],         "#0f172a"),
                      ("Relationships",s["total_relationships"],"#0d9488"),
                      ("DQ issues",    s["total_dq_issues"],
                       "#dc2626" if s["critical_dq"]>0 else "#ca8a04"),
                  ])
        + "</div>"
        + "<br><span style='font-size:.78rem;color:#475569'>"
          "💡 Try asking: <em>\"Show me age vs sex dashboard\"</em> · "
          "<em>\"What is the max age?\"</em> · "
          "<em>\"Show DQ issues\"</em> · "
          "<em>\"Show ER relationships\"</em></span>"
    )
    btns = (
        [("📊 Auto dashboard",   f"Show auto key column dashboard for {all_ds[0].name}"),
         ("🔗 Relationships",    "Show all relationships between datasets"),
         ("🛡️ DQ Issues",       "Show all data quality issues")]
        + [(f"📊 {d.name} dashboard", f"Dashboard for {d.name}") for d in all_ds[:4]]
        + [(f"📋 {d.name} schema",    f"Show schema of {d.name}") for d in all_ds[:4]]
    )
    st.session_state.messages.append({"role":"ai","html":html,"btns":btns,
                                       "sources":[],"charts":[]})

# ══════════════════════════════════════════════════════════════════════════════
# TOP BAR
# ══════════════════════════════════════════════════════════════════════════════
mode_map = {"welcome":"Ready","cdisc":"CDISC demo","dynamic":"Your data"}
n_chunks = (st.session_state.dyn_rag.stats()["total"]
            if st.session_state.dyn_rag else static_rag.stats()["total"])
st.markdown(f"""<div class="topbar">
  <span style='font-size:1.2rem'>🧬</span>
  <span class='t-title'>DataGenome AI</span>
  <span class='t-sub'>Upload · Analyse · Chat — No API key · Free</span>
  <div class='t-right'><span class='dot'></span>
  {mode_map.get(st.session_state.mode,'Active')} &nbsp;·&nbsp; {n_chunks} chunks</div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD BAR (always visible, compact single-line)
# ══════════════════════════════════════════════════════════════════════════════
with st.container():
    st.markdown('<div class="upload-bar">', unsafe_allow_html=True)
    ub_c1, ub_c2, ub_c3, ub_c4 = st.columns([4, 1.2, 1.2, 1])

    with ub_c1:
        uploaded = st.file_uploader(
            "📁 Upload CSV / Excel / JSON / SQLite (multiple OK)",
            type=["csv","tsv","txt","xlsx","xls","json","db","sqlite","sqlite3"],
            accept_multiple_files=True,
            label_visibility="visible",
            key="uploader",
        )

    with ub_c2:
        st.markdown("<div style='padding-top:1.6rem'>", unsafe_allow_html=True)
        st.markdown('<div class="analyse-wrap">', unsafe_allow_html=True)
        analyse_btn = st.button("🔍 Analyse", key="analyse_btn",
                                use_container_width=True,
                                disabled=not bool(uploaded))
        st.markdown("</div></div>", unsafe_allow_html=True)

    with ub_c3:
        st.markdown("<div style='padding-top:1.6rem'>", unsafe_allow_html=True)
        st.markdown('<div class="analyse-wrap">', unsafe_allow_html=True)
        cdisc_btn = st.button("🧬 CDISC Demo", key="cdisc_btn",
                              use_container_width=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    with ub_c4:
        st.markdown("<div style='padding-top:1.6rem'>", unsafe_allow_html=True)
        if st.session_state.datasets:
            st.markdown('<div class="clear-wrap">', unsafe_allow_html=True)
            if st.button("🗑️ Clear", key="clear_btn", use_container_width=True):
                for k in ["datasets","report","dyn_rag","messages"]:
                    st.session_state[k]=[] if k in ("datasets","messages") else None
                st.session_state.mode="welcome"
                st.session_state.input_key+=1
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Status row
    if st.session_state.datasets:
        pills = "".join(
            f'<span class="status-pill">✓ {d.name} ({d.row_count:,}r · {d.col_count}c)</span> '
            for d in st.session_state.datasets
        )
        s = st.session_state.report["summary"] if st.session_state.report else {}
        st.markdown(
            f'<div style="padding:.3rem 1.2rem;font-size:.73rem;'
            f'background:#f8fafc;border-bottom:1px solid #e9eef4">'
            f'{pills}'
            f'<span style="color:#64748b;margin-left:6px">'
            f'· {s.get("total_relationships",0)} relationships'
            f' · {s.get("total_dq_issues",0)} DQ issues</span></div>',
            unsafe_allow_html=True,
        )

# Handle analyse click
if analyse_btn and uploaded:
    with st.spinner("📊 Loading & profiling…"):
        all_ds, errors = [], []
        for f in uploaded:
            try:
                f.seek(0)
                all_ds.extend(load_uploaded_file(f))
            except Exception as e:
                errors.append(f"{f.name}: {e}")
    if errors:
        for err in errors: st.error(err)
    if all_ds:
        with st.spinner("🔗 Detecting relationships & DQ issues…"):
            _run_analysis(all_ds)
        st.rerun()

# Handle CDISC demo click
if cdisc_btn:
    with st.spinner("🧬 Loading CDISC demo…"):
        cdisc_ds = _load_cdisc_datasets()
        _run_analysis(cdisc_ds)
        st.session_state.mode = "cdisc"   # flag for static RAG fallback
    st.rerun()

st.markdown("<hr style='margin:0;border-color:#e9eef4'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# WELCOME SCREEN (shown only before first message)
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.messages:
    st.markdown("""
    <div style='text-align:center;padding:2.5rem 1rem 1.5rem;color:#64748b'>
      <div style='font-size:2.2rem;margin-bottom:.4rem'>🧬</div>
      <div style='font-size:.98rem;font-weight:800;color:#0f172a;margin-bottom:.4rem'>
        DataGenome AI</div>
      <div style='font-size:.82rem;line-height:1.7;max-width:520px;margin:0 auto .8rem;color:#475569'>
        Upload your CSV/Excel/JSON/SQLite files or try the CDISC demo above.<br>
        After analysis, ask anything — dashboards, statistics, schemas, relationships.
      </div>
      <div style='display:flex;flex-wrap:wrap;justify-content:center;gap:5px;max-width:600px;margin:0 auto'>
        <span style='background:#f1f5f9;border:1px solid #e2e8f0;border-radius:8px;
          padding:4px 10px;font-size:.74rem;color:#475569'>
          💡 Dashboard of age and sex</span>
        <span style='background:#f1f5f9;border:1px solid #e2e8f0;border-radius:8px;
          padding:4px 10px;font-size:.74rem;color:#475569'>
          💡 What is the max salary?</span>
        <span style='background:#f1f5f9;border:1px solid #e2e8f0;border-radius:8px;
          padding:4px 10px;font-size:.74rem;color:#475569'>
          💡 Show ER relationships</span>
        <span style='background:#f1f5f9;border:1px solid #e2e8f0;border-radius:8px;
          padding:4px 10px;font-size:.74rem;color:#475569'>
          💡 Compare ALT vs AST distribution</span>
        <span style='background:#f1f5f9;border:1px solid #e2e8f0;border-radius:8px;
          padding:4px 10px;font-size:.74rem;color:#475569'>
          💡 Show DQ issues</span>
      </div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CHAT MESSAGES
# ══════════════════════════════════════════════════════════════════════════════
chat_area = st.container()
with chat_area:
    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(
                f'<div class="bubble-user">👤 {msg["text"]}</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="bubble-ai">🧬 {msg["html"]}</div>',
                unsafe_allow_html=True)

            # ── Plotly charts embedded inline ──────────────────────────────
            charts = msg.get("charts", [])
            if charts:
                # Render in 2-column grid
                pairs = [charts[i:i+2] for i in range(0, len(charts), 2)]
                for pair in pairs:
                    cols = st.columns(len(pair))
                    for ci, (ctitle, fig) in enumerate(pair):
                        with cols[ci]:
                            st.plotly_chart(fig, use_container_width=True,
                                            config={"displayModeBar": False})

            # Sources
            sources = msg.get("sources", [])
            if sources:
                def _chip(s):
                    cat   = s.get("category","info")
                    score = s.get("score","—")
                    title = s.get("title","")
                    short = title[:40]+("…" if len(title)>40 else "")
                    return f'<span class="src-chip {cat}" title="{score}">{short}</span>'
                chips = "".join(_chip(s) for s in sources[:5])
                st.markdown(f'<div class="sources">📎 {chips}</div>',
                            unsafe_allow_html=True)

            # Action buttons
            btns = msg.get("btns", [])
            if btns:
                n  = min(len(btns), 4)
                bc = st.columns(n)
                for j,(lbl,q) in enumerate(btns):
                    with bc[j%n]:
                        if st.button(lbl, key=f"b_{idx}_{j}"):
                            st.session_state.pending=q
                            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# INPUT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr style='margin:.2rem 0 0;border-color:#e9eef4'>", unsafe_allow_html=True)
in_c, btn_c = st.columns([6,1])
with in_c:
    loaded = bool(st.session_state.datasets)
    ph = ("Ask anything… e.g. 'Dashboard of AGE and SEX' · 'Max salary' · "
          "'Show DQ issues' · 'ER diagram' · 'Schema of patients'"
          if loaded else
          "Upload files above or try CDISC Demo, then ask anything…")
    user_input = st.text_area("msg", label_visibility="collapsed",
                               placeholder=ph, height=66,
                               key=f"inp_{st.session_state.input_key}")
with btn_c:
    st.markdown("<div class='send-wrap' style='padding-top:4px'>", unsafe_allow_html=True)
    send = st.button("Send ▶", key="send_btn", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

if send and user_input and user_input.strip():
    submit(user_input)
    st.rerun()

st.markdown(
    "<div style='text-align:center;font-size:.63rem;color:#cbd5e1;padding:.2rem 0 .35rem'>"
    "BM25 retrieval · Plotly charts · pandas profiling · Free · No API key"
    "</div>", unsafe_allow_html=True)
