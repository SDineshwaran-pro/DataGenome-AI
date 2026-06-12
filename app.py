import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data import CLINICAL_TABLES, GLOSSARY, MOCK_DQ_ISSUES

st.set_page_config(
    page_title="DataGenome AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { 
        background: linear-gradient(135deg, #0f172a 0%, #134e4a 100%);
        color: white; padding: 1.2rem 1.5rem; border-radius: 12px;
        margin-bottom: 1.2rem;
    }
    .main-header h1 { margin: 0; font-size: 1.6rem; font-weight: 800; }
    .main-header p { margin: 0.2rem 0 0; opacity: 0.7; font-size: 0.85rem; }
    .kpi-card {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: 1rem; text-align: center;
    }
    .kpi-number { font-size: 1.8rem; font-weight: 900; color: #0f172a; font-family: monospace; }
    .kpi-label { font-size: 0.75rem; color: #64748b; margin-top: 0.2rem; }
    .tag-sdtm { background: #ccfbf1; color: #0f766e; padding: 2px 8px; border-radius: 20px; font-size: 0.7rem; font-weight: 700; }
    .tag-adam { background: #dbeafe; color: #1d4ed8; padding: 2px 8px; border-radius: 20px; font-size: 0.7rem; font-weight: 700; }
    .badge-critical { background: #fee2e2; color: #dc2626; padding: 2px 8px; border-radius: 20px; font-size: 0.7rem; font-weight: 600; }
    .badge-warning { background: #fef9c3; color: #ca8a04; padding: 2px 8px; border-radius: 20px; font-size: 0.7rem; font-weight: 600; }
    .badge-info { background: #dbeafe; color: #2563eb; padding: 2px 8px; border-radius: 20px; font-size: 0.7rem; font-weight: 600; }
    .chat-user { background: #0d9488; color: white; padding: 0.7rem 1rem; border-radius: 12px 12px 2px 12px; margin: 0.4rem 0; }
    .chat-ai { background: #f1f5f9; color: #1e293b; padding: 0.7rem 1rem; border-radius: 2px 12px 12px 12px; margin: 0.4rem 0; border-left: 3px solid #0d9488; }
    div[data-testid="stSidebar"] { background: #0f172a !important; }
    div[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
    div[data-testid="stSidebar"] .stRadio label { color: #cbd5e1 !important; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
  <h1>🧬 DataGenome AI</h1>
  <p>Conversational Clinical Data Intelligence · CDISC SDTM & ADaM · Zero SQL Required</p>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.markdown("## 🧬 DataGenome AI")
    st.markdown("---")
    view = st.radio(
        "Navigate",
        options=[
            "💬 AI Chatbot",
            "📋 Schema Explorer",
            "🔗 ER Diagram",
            "🛡️ DQ Audit",
            "📊 DQ Dashboard",
            "📖 Data Dictionary",
            "📜 Regulatory Report",
            "📚 GCP Glossary",
        ],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Study:** STUDY-2026-FIBER")
    st.markdown("**Records:** 10,240 total")
    st.markdown("**Tables:** 5 (SDTM + ADaM)")
    st.caption("Free prototype · No API key required")

# ─── CHATBOT ───────────────────────────────────────────────────────────────────
if view == "💬 AI Chatbot":
    st.subheader("💬 Clinical Intelligence Assistant")
    st.caption("Ask questions about your clinical database in plain English.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Welcome to DataGenome AI! I'm your Conversational Clinical Data Intelligence Agent.\n\nNo SQL knowledge needed. Ask me about schemas, adverse events, data quality issues, ER relationships, or regulatory dossier generation.\n\nTry: *\"Show me adverse event severity breakdown\"* or *\"What are the DQ issues in the DM table?\"*"}
        ]

    # Quick action chips
    st.markdown("**Quick Actions:**")
    cols = st.columns(4)
    quick_actions = [
        "Show DQ audit issues",
        "What tables are in SDTM?",
        "Explain USUBJID field",
        "Show ER relationships",
    ]
    for i, action in enumerate(quick_actions):
        if cols[i].button(action, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": action})
            st.rerun()

    # Chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-ai">🧬 {msg["content"]}</div>', unsafe_allow_html=True)

    # Generate rule-based response (no API needed)
    last = st.session_state.messages[-1] if st.session_state.messages else None
    if last and last["role"] == "user":
        q = last["content"].lower()
        if any(w in q for w in ["dq", "quality", "audit", "issue", "null", "missing", "anomaly"]):
            response = (
                "📊 **DQ Audit Summary**\n\n"
                "I found **6 active constraint exceptions** in your clinical database:\n\n"
                "• **DM.ARMCD** — 7 subjects with blank arm assignment (Critical)\n"
                "• **VS.VSSTRESN** — 3 negative systolic BP records (Critical)\n"
                "• **AE.AESTDTC** — 2 events with future dates beyond 2026 (Warning)\n"
                "• **LB.LBSTRESN** — 12 out-of-range lab values (Warning)\n\n"
                "Recommended: navigate to 🛡️ DQ Audit for full remediation guidance."
            )
        elif any(w in q for w in ["table", "schema", "sdtm", "adam", "structure"]):
            response = (
                "📋 **Connected Clinical Tables**\n\n"
                "Your study contains **5 CDISC-compliant tables**:\n\n"
                "• **DM** (Demographics) — 250 rows · SDTM\n"
                "• **AE** (Adverse Events) — 1,140 rows · SDTM\n"
                "• **VS** (Vital Signs) — 3,200 rows · SDTM\n"
                "• **LB** (Laboratory Results) — 5,400 rows · SDTM\n"
                "• **ADSL** (Subject-Level Analysis) — 250 rows · ADaM\n\n"
                "Navigate to 📋 Schema Explorer to inspect columns and sample data."
            )
        elif any(w in q for w in ["er", "relation", "foreign", "link", "connect"]):
            response = (
                "🔗 **ER Relationship Summary**\n\n"
                "All 5 tables share **USUBJID** (Unique Subject Identifier) as the primary link:\n\n"
                "• **DM → AE**: 1:many via USUBJID + STUDYID\n"
                "• **DM → VS**: 1:many via USUBJID + STUDYID\n"
                "• **DM → LB**: 1:many via USUBJID + STUDYID\n"
                "• **DM → ADSL**: 1:1 via USUBJID\n\n"
                "Navigate to 🔗 ER Diagram for the visual relationship map."
            )
        elif any(w in q for w in ["usubjid", "studyid", "armcd", "aesev", "field", "column", "variable"]):
            response = (
                "📖 **Field Definition**\n\n"
                "**USUBJID** — Unique Subject Identifier\n"
                "Standard: CDISC SDTM | Type: VARCHAR(100) | Required: Yes\n\n"
                "A globally unique identifier combining sponsor, study, and subject number (e.g., `01-10023`). "
                "This is the primary linkage key across all CDISC tables and must never be null.\n\n"
                "Navigate to 📖 Data Dictionary for all 40+ field definitions."
            )
        elif any(w in q for w in ["adverse", "ae", "event", "severity", "aesev"]):
            response = (
                "⚠️ **Adverse Events Summary**\n\n"
                "Your AE domain contains **1,140 records** across all subjects.\n\n"
                "Severity breakdown:\n"
                "• MILD — 624 events (54.7%)\n"
                "• MODERATE — 398 events (34.9%)\n"
                "• SEVERE — 118 events (10.4%)\n\n"
                "Serious adverse events (AESER = 'Y'): **47 records**\n"
                "Most common term: Upper Respiratory Tract Infection\n\n"
                "Navigate to 📊 DQ Dashboard for trends over time."
            )
        elif any(w in q for w in ["glossary", "meddra", "gcp", "alcoa", "term", "definition", "mean"]):
            response = (
                "📚 **Clinical Term Lookup**\n\n"
                "Key terms in your study:\n\n"
                "• **MedDRA** — Medical Dictionary for Regulatory Activities. Hierarchical coding system for adverse events.\n"
                "• **SDTM** — Study Data Tabulation Model. CDISC standard for raw clinical trial data.\n"
                "• **ADaM** — Analysis Data Model. Derived datasets ready for statistical analysis.\n"
                "• **ALCOA+** — Attributable, Legible, Contemporaneous, Original, Accurate + Complete, Consistent, Enduring, Available.\n\n"
                "Navigate to 📚 GCP Glossary for 20+ clinical definitions."
            )
        else:
            response = (
                "🧬 I understand you're asking about your clinical database. "
                "I can help you with:\n\n"
                "• **Schema exploration** — table structures, column definitions\n"
                "• **Data quality** — anomalies, constraint violations, remediation\n"
                "• **ER relationships** — how tables connect via USUBJID\n"
                "• **Regulatory reports** — CDISC-compliant documentation\n"
                "• **Glossary** — GCP, MedDRA, SDTM, ADaM terminology\n\n"
                "Try asking: *\"What DQ issues exist?\"* or *\"Show me the AE table schema.\"*"
            )
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        user_input = col1.text_input("Ask a question...", label_visibility="collapsed", placeholder="e.g. What are the DQ issues in the VS table?")
        submitted = col2.form_submit_button("Send", use_container_width=True)
        if submitted and user_input.strip():
            st.session_state.messages.append({"role": "user", "content": user_input.strip()})
            st.rerun()

# ─── SCHEMA EXPLORER ───────────────────────────────────────────────────────────
elif view == "📋 Schema Explorer":
    st.subheader("📋 Clinical Schema Intelligence")
    st.caption("Auto-discovered table indices and column categorization per CDISC SDTM & ADaM frameworks.")

    col1, col2 = st.columns([1, 3])

    with col1:
        table_options = {t["name"]: t for t in CLINICAL_TABLES}
        selected_name = st.radio("Select Table", list(table_options.keys()))
        table = table_options[selected_name]
        badge = "tag-sdtm" if table["standard"] == "SDTM" else "tag-adam"
        st.markdown(f"""
        <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:0.8rem;margin-top:0.5rem'>
            <div style='font-weight:800;font-size:1.1rem'>{table['name']}</div>
            <span class='{badge}'>{table['standard']}</span>
            <div style='font-size:0.75rem;color:#64748b;margin-top:0.4rem'>{table['label']}</div>
            <div style='font-size:0.8rem;color:#475569;margin-top:0.4rem'>{table['description']}</div>
            <div style='font-size:0.75rem;font-weight:700;color:#0d9488;margin-top:0.5rem'>{table['rowCount']:,} rows</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        search = st.text_input("🔍 Search columns", placeholder="column name, type, description...")
        cols_data = table["columns"]
        if search:
            cols_data = [c for c in cols_data if
                         search.lower() in c["name"].lower() or
                         search.lower() in c["description"].lower() or
                         search.lower() in c["type"].lower()]

        rows = []
        for c in cols_data:
            flags = []
            if c["isPrimary"]: flags.append("🔑 PK")
            if c["isForeign"]: flags.append("🔗 FK")
            rows.append({
                "Column": c["name"],
                "Type": c["type"],
                "Standard": c["cdiscStandard"],
                "Flags": " ".join(flags) if flags else "—",
                "Nullable": "Yes" if c["nullable"] else "No",
                "Description": c["description"],
                "Sample Data": ", ".join(c["sampleData"][:2]),
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True, height=420)
        st.caption(f"Showing {len(rows)} of {len(table['columns'])} columns")

# ─── ER DIAGRAM ────────────────────────────────────────────────────────────────
elif view == "🔗 ER Diagram":
    st.subheader("🔗 Entity-Relationship Diagram")
    st.caption("Visual map of CDISC table relationships via shared subject identifiers.")

    fig = go.Figure()

    # Node positions
    nodes = {
        "DM": (0.5, 0.8, "#0d9488", "Demographics\n250 rows · SDTM"),
        "AE": (0.1, 0.4, "#f43f5e", "Adverse Events\n1,140 rows · SDTM"),
        "VS": (0.35, 0.15, "#eab308", "Vital Signs\n3,200 rows · SDTM"),
        "LB": (0.65, 0.15, "#3b82f6", "Laboratory\n5,400 rows · SDTM"),
        "ADSL": (0.9, 0.4, "#6366f1", "Subj-Level Analysis\n250 rows · ADaM"),
    }

    edges = [("DM", "AE"), ("DM", "VS"), ("DM", "LB"), ("DM", "ADSL")]
    for src, tgt in edges:
        x0, y0 = nodes[src][:2]
        x1, y1 = nodes[tgt][:2]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(color="#94a3b8", width=2.5, dash="dot"),
            hoverinfo="skip", showlegend=False
        ))
        # USUBJID label on edge
        fig.add_annotation(
            x=(x0+x1)/2, y=(y0+y1)/2,
            text="USUBJID", showarrow=False,
            font=dict(size=9, color="#64748b"),
            bgcolor="white", borderpad=2
        )

    for name, (x, y, color, label) in nodes.items():
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(size=60, color=color, line=dict(color="white", width=3)),
            text=[name], textposition="middle center",
            textfont=dict(color="white", size=14, family="monospace"),
            hovertext=[label], hoverinfo="text",
            showlegend=False
        ))

    fig.update_layout(
        height=460, margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.05, 1.05]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.05, 1.05]),
        plot_bgcolor="white", paper_bgcolor="white"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Relationship Legend**")
    rel_df = pd.DataFrame([
        {"From": "DM", "To": "AE", "Type": "1:Many", "Key": "USUBJID + STUDYID", "Rule": "One subject → many adverse events"},
        {"From": "DM", "To": "VS", "Type": "1:Many", "Key": "USUBJID + STUDYID", "Rule": "One subject → many vital sign timepoints"},
        {"From": "DM", "To": "LB", "Type": "1:Many", "Key": "USUBJID + STUDYID", "Rule": "One subject → many lab results"},
        {"From": "DM", "To": "ADSL", "Type": "1:1", "Key": "USUBJID", "Rule": "One subject → one analysis record"},
    ])
    st.dataframe(rel_df, use_container_width=True, hide_index=True)

# ─── DQ AUDIT ──────────────────────────────────────────────────────────────────
elif view == "🛡️ DQ Audit":
    st.subheader("🛡️ Data Quality Diagnostics")
    st.caption("Automated constraint violation detection across CDISC domains.")

    # KPIs
    critical = sum(1 for i in MOCK_DQ_ISSUES if i["severity"] == "Critical")
    warning = sum(1 for i in MOCK_DQ_ISSUES if i["severity"] == "Warning")
    info = sum(1 for i in MOCK_DQ_ISSUES if i["severity"] == "Info")
    total_affected = sum(i["count"] for i in MOCK_DQ_ISSUES)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🔴 Critical", critical, "Blockers")
    k2.metric("🟡 Warnings", warning, "Review needed")
    k3.metric("🔵 Info", info, "Advisory")
    k4.metric("📊 Affected Rows", total_affected)

    st.divider()

    # Filters
    f1, f2 = st.columns(2)
    sev_filter = f1.selectbox("Severity", ["All", "Critical", "Warning", "Info"])
    table_filter = f2.selectbox("Table", ["All"] + list({i["tableId"].upper() for i in MOCK_DQ_ISSUES}))

    issues = MOCK_DQ_ISSUES
    if sev_filter != "All":
        issues = [i for i in issues if i["severity"] == sev_filter]
    if table_filter != "All":
        issues = [i for i in issues if i["tableId"].upper() == table_filter]

    for issue in issues:
        sev_color = {"Critical": "🔴", "Warning": "🟡", "Info": "🔵"}.get(issue["severity"], "⚪")
        with st.expander(f"{sev_color} [{issue['tableId'].upper()}] {issue['columnName']} — {issue['issueType']} ({issue['count']} records)"):
            c1, c2 = st.columns(2)
            c1.markdown(f"**Issue:** {issue['description']}")
            c1.markdown(f"**Type:** `{issue['issueType']}`")
            c1.markdown(f"**Affected:** {issue['count']} rows ({issue['percentage']}%)")
            c2.info(f"**Remediation:** {issue['remediation']}")

# ─── DQ DASHBOARD ──────────────────────────────────────────────────────────────
elif view == "📊 DQ Dashboard":
    st.subheader("📊 Clinical Operations DQ Dashboard")
    st.caption("Real-time analytics: completeness metrics, error trends, and exception profiles.")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🗃️ Total Records", "10,240")
    k2.metric("⚠️ Exceptions", "6")
    k3.metric("✅ Avg Completeness", "99.2%")
    k4.metric("📉 Error Reduction", "↓87% (6mo)")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        completeness_data = pd.DataFrame([
            {"Table": "DM (Demog)", "Completeness %": 98.4},
            {"Table": "AE (Events)", "Completeness %": 99.9},
            {"Table": "VS (Vitals)", "Completeness %": 99.8},
            {"Table": "LB (Labs)", "Completeness %": 98.4},
            {"Table": "ADSL (Analysis)", "Completeness %": 99.6},
        ])
        fig1 = px.bar(completeness_data, x="Completeness %", y="Table", orientation="h",
                      color="Completeness %", color_continuous_scale="Teal",
                      range_x=[97, 100.2], title="Data Completeness by Table (%)")
        fig1.update_layout(height=280, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        sev_data = pd.DataFrame([
            {"Severity": "Critical Blockers", "Count": 17},
            {"Severity": "Safety Warnings", "Count": 152},
            {"Severity": "Advisory Logs", "Count": 54},
        ])
        fig2 = px.pie(sev_data, names="Severity", values="Count",
                      color_discrete_sequence=["#ef4444", "#f59e0b", "#3b82f6"],
                      title="Exception Severity Distribution")
        fig2.update_layout(height=280, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    timeline_data = pd.DataFrame([
        {"Month": "Jan 26", "Anomalies": 45, "Error Rate %": 5.2},
        {"Month": "Feb 26", "Anomalies": 38, "Error Rate %": 4.1},
        {"Month": "Mar 26", "Anomalies": 27, "Error Rate %": 3.2},
        {"Month": "Apr 26", "Anomalies": 19, "Error Rate %": 2.1},
        {"Month": "May 26", "Anomalies": 8, "Error Rate %": 0.9},
        {"Month": "Jun 26", "Anomalies": 6, "Error Rate %": 0.6},
    ])
    fig3 = px.line(timeline_data, x="Month", y=["Anomalies", "Error Rate %"],
                   title="Anomaly Reduction Trend (2026)", markers=True,
                   color_discrete_sequence=["#0d9488", "#f43f5e"])
    fig3.update_layout(height=280, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig3, use_container_width=True)

# ─── DATA DICTIONARY ───────────────────────────────────────────────────────────
elif view == "📖 Data Dictionary":
    st.subheader("📖 Clinical Data Dictionary")
    st.caption("Comprehensive column definitions, data types, and CDISC standard classifications.")

    all_cols = []
    for table in CLINICAL_TABLES:
        for col in table["columns"]:
            all_cols.append({
                "Table": table["name"],
                "Standard": table["standard"],
                "Column": col["name"],
                "Type": col["type"],
                "CDISC": col["cdiscStandard"],
                "PK": "✅" if col["isPrimary"] else "",
                "FK": "✅" if col["isForeign"] else "",
                "Nullable": "Yes" if col["nullable"] else "No",
                "Description": col["description"],
                "Sample": ", ".join(col["sampleData"][:2]),
            })

    df = pd.DataFrame(all_cols)

    c1, c2, c3 = st.columns(3)
    table_f = c1.selectbox("Filter Table", ["All"] + list(df["Table"].unique()))
    std_f = c2.selectbox("Filter Standard", ["All", "SDTM", "ADaM", "Both", "General"])
    search_f = c3.text_input("🔍 Search column/description")

    if table_f != "All":
        df = df[df["Table"] == table_f]
    if std_f != "All":
        df = df[df["CDISC"] == std_f]
    if search_f:
        mask = (df["Column"].str.contains(search_f, case=False) |
                df["Description"].str.contains(search_f, case=False))
        df = df[mask]

    st.dataframe(df, use_container_width=True, hide_index=True, height=480)
    st.caption(f"Showing {len(df)} columns")

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Export CSV", csv, "data_dictionary.csv", "text/csv")

# ─── REGULATORY REPORT ─────────────────────────────────────────────────────────
elif view == "📜 Regulatory Report":
    st.subheader("📜 Regulatory Dossier Generator")
    st.caption("CDISC-compliant documentation for FDA/EMA submission review.")

    st.success("✅ Study STUDY-2026-FIBER · Report generated: June 12, 2026")

    tab1, tab2, tab3 = st.tabs(["📝 Study Summary", "🗂️ Dataset Inventory", "📋 DQ Compliance"])

    with tab1:
        st.markdown("""
### Study Synopsis

| Field | Value |
|---|---|
| **Study ID** | STUDY-2026-FIBER |
| **Protocol Title** | Phase III Randomized Clinical Trial — Drug X vs Placebo |
| **Sponsor** | DataGenome Pharma Inc. |
| **Phase** | III |
| **Indication** | Chronic Inflammatory Disease |
| **Primary Endpoint** | Reduction in disease severity score at Week 24 |
| **Randomization** | 1:1 Active vs Placebo |
| **Subjects Enrolled** | 250 |
| **CDISC Standard** | SDTM 3.3 + ADaM 1.3 |
| **Data Cutoff** | May 31, 2026 |
""")

    with tab2:
        inv_df = pd.DataFrame([
            {"Domain": t["name"], "Label": t["label"], "Standard": t["standard"],
             "Records": t["rowCount"], "Status": "✅ Validated"}
            for t in CLINICAL_TABLES
        ])
        st.dataframe(inv_df, use_container_width=True, hide_index=True)
        st.markdown(f"**Total Records:** {sum(t['rowCount'] for t in CLINICAL_TABLES):,}")

    with tab3:
        dq_df = pd.DataFrame([
            {
                "Domain": i["tableId"].upper(),
                "Variable": i["columnName"],
                "Issue Type": i["issueType"],
                "Count": i["count"],
                "% Affected": f"{i['percentage']}%",
                "Severity": i["severity"],
                "Status": "🔴 Open" if i["severity"] == "Critical" else "🟡 Under Review",
            }
            for i in MOCK_DQ_ISSUES
        ])
        st.dataframe(dq_df, use_container_width=True, hide_index=True)

        csv = dq_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export DQ Report CSV", csv, "dq_compliance_report.csv", "text/csv")

# ─── GCP GLOSSARY ──────────────────────────────────────────────────────────────
elif view == "📚 GCP Glossary":
    st.subheader("📚 Clinical Regulatory Glossary")
    st.caption("Life sciences jargon, CDISC acronyms, MedDRA terms, and GCP guidelines.")

    c1, c2 = st.columns([2, 1])
    search_g = c1.text_input("🔍 Search terms and definitions")
    std_g = c2.selectbox("Standard", ["ALL", "MedDRA", "SDTM", "ADaM", "GCP", "RWE"])

    items = GLOSSARY
    if std_g != "ALL":
        items = [g for g in items if g["standard"] == std_g]
    if search_g:
        items = [g for g in items if
                 search_g.lower() in g["term"].lower() or
                 search_g.lower() in g["definition"].lower()]

    for g in items:
        color_map = {"MedDRA": "#fce7f3", "SDTM": "#ccfbf1", "ADaM": "#dbeafe", "GCP": "#fef3c7", "RWE": "#f3e8ff"}
        bg = color_map.get(g["standard"], "#f8fafc")
        with st.expander(f"**{g['term']}** · {g['standard']} · {g['category']}"):
            st.markdown(f"**Definition:** {g['definition']}")
            st.markdown(f"**Example:** _{g['example']}_")
            if g.get("code"):
                st.code(g["code"], language="text")

    st.caption(f"Showing {len(items)} terms")
