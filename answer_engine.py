"""
answer_engine.py  —  RAG-driven answer synthesis
Converts retrieved BM25 chunks into rich, structured answers.
No LLM API required — fully deterministic, data-driven responses.
"""

from data import CLINICAL_TABLES, GLOSSARY, MOCK_DQ_ISSUES

# ─── HTML helpers ─────────────────────────────────────────────────────────────
def _stag(std):
    cls = "ts" if std == "SDTM" else "ta"
    return f'<span class="tag {cls}">{std}</span>'

def _sevtag(sev):
    cls = {"Critical": "tc", "Warning": "tw", "Info": "ti"}.get(sev, "ti")
    return f'<span class="tag {cls}">{sev}</span>'

def _card(label, value, color=None):
    style = f'style="color:{color}"' if color else ""
    return (f'<div class="sc"><div class="sc-l">{label}</div>'
            f'<div class="sc-v" {style}>{value}</div></div>')

def _cards(*items):
    return '<div class="sc-row">' + "".join(_card(*i) for i in items) + "</div>"

def _bar(label, pct, color="#0d9488", note="", width=68):
    return (
        f'<div class="bar-row">'
        f'<span style="width:{width}px;font-weight:500;font-size:0.76rem">{label}</span>'
        f'<div class="bar-bg"><div class="bar-fill" style="width:{min(pct,100):.1f}%;background:{color}"></div></div>'
        f'<span style="width:80px;text-align:right;color:#64748b;font-size:0.74rem">{note}</span>'
        f'</div>'
    )

def _table(headers, rows, max_h=None):
    style = f'style="max-height:{max_h};overflow-y:auto"' if max_h else ""
    ths = "".join(
        f'<th style="padding:6px 10px;text-align:left;font-size:0.7rem;color:#64748b;'
        f'border-bottom:1px solid #e2e8f0;background:#f8fafc;white-space:nowrap">{h}</th>'
        for h in headers
    )
    trs = "".join(
        "<tr>" + "".join(
            f'<td style="padding:5px 10px;font-size:0.75rem;border-bottom:1px solid #f1f5f9;'
            f'vertical-align:top">{c}</td>'
            for c in row
        ) + "</tr>"
        for row in rows
    )
    return (
        f'<div style="border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;margin-top:6px;{style}">'
        f'<table style="width:100%;border-collapse:collapse">'
        f'<thead><tr>{ths}</tr></thead><tbody>{trs}</tbody></table></div>'
    )

def _section(title):
    return (f'<div style="margin:10px 0 4px;font-size:0.68rem;font-weight:700;'
            f'color:#94a3b8;text-transform:uppercase;letter-spacing:.06em">{title}</div>')

def _opt_note(text):
    return f'<div style="font-size:0.73rem;color:#94a3b8;margin-top:6px">{text}</div>'


# ─── Lookup helpers ───────────────────────────────────────────────────────────
_TABLE_MAP = {t["name"]: t for t in CLINICAL_TABLES}
_ALL_COLS  = [(t["name"], t["standard"], c) for t in CLINICAL_TABLES for c in t["columns"]]
_COL_MAP   = {(t["name"], c["name"]): c for t, _, c in [(t, t["standard"], c)
               for t in CLINICAL_TABLES for c in t["columns"]]}
_GLOS_MAP  = {g["term"].lower(): g for g in GLOSSARY}


# ─── Intent detection ─────────────────────────────────────────────────────────
def _detect_intent(q: str, chunks: list) -> str:
    ql = q.lower()
    cats = [c["category"] for c in chunks]

    # Explicit keyword signals
    if any(w in ql for w in ["help", "menu", "option", "start", "what can", "hi ", "hello", "hey "]) and len(q) < 55:
        return "help"
    if any(w in ql for w in ["dq dashboard", "completeness", "trend", "metric", "kpi", "error rate"]):
        return "dq_dashboard"
    if any(w in ql for w in ["dq", "data quality", "audit", "issue", "violation", "anomaly",
                               "missing", "duplicate", "negative value", "future date", "null"]):
        return "dq_audit"
    if any(w in ql for w in ["er diagram", "er relation", "entity relation", "foreign key",
                               "join", "link between", "relationship"]):
        return "er"
    if any(w in ql for w in ["data dictionary", "all column", "all field", "full schema",
                               "all variable", "dictionary"]):
        return "dictionary"
    if any(w in ql for w in ["glossary", "all term", "all definition", "list term"]):
        return "glossary_all"
    if any(w in ql for w in ["regulatory", "fda", "ema", "submission", "dossier",
                               "pinnacle", "define.xml", "compliance check"]):
        return "regulatory"
    if any(w in ql for w in ["adverse event", "ae summary", "ae table", "aesev",
                               "serious", "mild", "moderate", "severe event"]):
        return "ae_summary"
    if any(w in ql for w in ["population", "itt", "saffl", "ittfl", "safety population",
                               "intent to treat", "per protocol", "analysis set"]):
        return "populations"
    if any(w in ql for w in ["study overview", "study summary", "phase", "sponsor",
                               "indication", "endpoint", "what is this study"]):
        return "study"

    # Table-specific column query
    for t in CLINICAL_TABLES:
        if t["name"].lower() in ql and any(w in ql for w in
           ["column", "field", "schema", "variable", "structure", "show", "describe", "detail"]):
            return f"table_schema:{t['name']}"

    # All-tables schema list
    if any(w in ql for w in ["table", "schema", "sdtm", "adam", "domain",
                               "list table", "all table", "what table"]):
        return "schema_list"

    # Specific column lookup — any column name in query
    for tn, std, c in _ALL_COLS:
        if c["name"].lower() in ql and len(c["name"]) > 3:
            return f"column:{tn}:{c['name']}"

    # Glossary term lookup
    for term, g in _GLOS_MAP.items():
        if term in ql or (len(term) > 3 and term in ql):
            return f"glossary_term:{g['term']}"

    if any(w in ql for w in ["what is", "define", "explain", "meaning", "mean"]):
        # Check if it's asking about a glossary term
        for term, g in _GLOS_MAP.items():
            if term in ql:
                return f"glossary_term:{g['term']}"

    # Fallback: use top chunk category
    if cats:
        top_cat = cats[0]
        if top_cat == "dq":        return "dq_audit"
        if top_cat == "schema":    return "schema_list"
        if top_cat == "column":    return "schema_list"
        if top_cat == "glossary":  return "glossary_all"
        if top_cat == "er":        return "er"
        if top_cat == "regulatory":return "regulatory"
        if top_cat == "ae":        return "ae_summary"

    return "fallback"


# ─── Answer builders ──────────────────────────────────────────────────────────

def _ans_help():
    return (
        "<strong>👋 Welcome to DataGenome AI</strong> — "
        "RAG-powered clinical intelligence for <code>STUDY-2026-FIBER</code>.<br>"
        "<span style='color:#64748b;font-size:0.81rem'>"
        "76 knowledge chunks · BM25 retrieval · Zero API key required"
        "</span>"
    ), [
        ("📋 Schema Explorer",       "List all SDTM and ADaM tables"),
        ("🛡️ DQ Audit",              "Show all data quality issues and remediations"),
        ("📊 DQ Dashboard",          "Show DQ completeness metrics and error trends"),
        ("🔗 ER Relationships",      "Explain ER relationships between all tables"),
        ("📖 Data Dictionary",       "Show full data dictionary for all columns"),
        ("📚 GCP Glossary",          "List all CDISC and GCP glossary terms"),
        ("📜 Regulatory Report",     "Show regulatory submission status and dossier"),
        ("⚠️ Adverse Events",        "Summarise AE domain data and severity breakdown"),
    ]


def _ans_schema_list():
    rows = [
        [f'<strong>{t["name"]}</strong>', _stag(t["standard"]),
         t["label"], f'{t["rowCount"]:,}', str(len(t["columns"])),
         t["description"][:60] + "…"]
        for t in CLINICAL_TABLES
    ]
    html = (
        "<strong>📋 CDISC Clinical Tables — STUDY-2026-FIBER</strong><br>"
        + _cards(
            ("Total tables", len(CLINICAL_TABLES), None),
            ("Total records", f'{sum(t["rowCount"] for t in CLINICAL_TABLES):,}', None),
            ("SDTM domains", sum(1 for t in CLINICAL_TABLES if t["standard"]=="SDTM"), None),
            ("ADaM datasets", sum(1 for t in CLINICAL_TABLES if t["standard"]=="ADaM"), None),
        )
        + _table(["Table","Std","Label","Rows","Cols","Description"], rows)
        + _opt_note('Ask: "Show me the AE table schema" or "What columns does LB have?"')
    )
    btns = [(f"📋 {t['name']} — {t['label']}", f"Describe the {t['name']} table columns") for t in CLINICAL_TABLES]
    return html, btns


def _ans_table_schema(tname: str):
    t = _TABLE_MAP.get(tname)
    if not t:
        return _ans_fallback()
    rows = []
    for c in t["columns"]:
        flags = ("🔑 " if c["isPrimary"] else "") + ("🔗" if c.get("isForeign") else "")
        fk_note = (f'→ {c["foreignTable"]}.{c["foreignColumn"]}'
                   if c.get("isForeign") and c.get("foreignTable") else "")
        rows.append([
            f'<code style="color:#0f766e">{c["name"]}</code>',
            f'<code style="color:#64748b;font-size:0.69rem">{c["type"]}</code>',
            flags or "—",
            "✓" if not c["nullable"] else "○",
            f'{c["description"]}{(" " + fk_note) if fk_note else ""}',
            f'<code style="font-size:0.69rem">{c["sampleData"][0]}</code>',
        ])
    html = (
        f'<strong>📋 {t["name"]} — {t["label"]}</strong> {_stag(t["standard"])}<br>'
        f'<span style="font-size:0.77rem;color:#64748b">{t["description"]}</span>'
        + _cards(
            ("Rows", f'{t["rowCount"]:,}', None),
            ("Columns", len(t["columns"]), None),
            ("Primary keys", sum(1 for c in t["columns"] if c["isPrimary"]), None),
            ("Foreign keys", sum(1 for c in t["columns"] if c.get("isForeign")), None),
        )
        + _table(["Column","Type","PK/FK","Req","Description","Sample"], rows)
        + _opt_note('✓ = Required field &nbsp;|&nbsp; 🔑 = Primary key &nbsp;|&nbsp; 🔗 = Foreign key')
    )
    return html, [("← All tables", "List all SDTM and ADaM tables")]


def _ans_dq_audit(query: str):
    ql = query.lower()
    # Filter by table if mentioned
    tf = [i for i in MOCK_DQ_ISSUES if i["tableId"].upper() in ql.upper()]
    display = tf if tf else MOCK_DQ_ISSUES
    crit  = [i for i in MOCK_DQ_ISSUES if i["severity"] == "Critical"]
    warns = [i for i in MOCK_DQ_ISSUES if i["severity"] == "Warning"]
    total_affected = sum(i["count"] for i in MOCK_DQ_ISSUES)

    rows = [
        [_sevtag(i["severity"]),
         f'<strong>{i["tableId"].upper()}</strong>',
         f'<code>{i["columnName"]}</code>',
         i["issueType"], str(i["count"]), f'{i["percentage"]}%']
        for i in display
    ]
    remeds = "".join(
        f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:7px;'
        f'padding:7px 10px;margin-top:5px;font-size:0.76rem">'
        f'{_sevtag(i["severity"])} '
        f'<strong>{i["tableId"].upper()}.{i["columnName"]}</strong> '
        f'— {i["issueType"]}<br>'
        f'<span style="color:#475569">{i["description"]}</span><br>'
        f'<span style="color:#0d9488;font-weight:600">▶ Fix: </span>'
        f'<span style="color:#0d9488">{i["remediation"]}</span></div>'
        for i in display
    )
    title = (f'<strong>🛡️ DQ Issues — {display[0]["tableId"].upper()} Table</strong>'
             if tf else f'<strong>🛡️ DQ Audit — {len(MOCK_DQ_ISSUES)} Active Issues</strong>')
    stats_html = "" if tf else _cards(
        ("Critical", len(crit), "#dc2626"),
        ("Warnings", len(warns), "#ca8a04"),
        ("Affected rows", total_affected, None),
        ("Tables affected", len({i["tableId"] for i in MOCK_DQ_ISSUES}), None),
    )
    html = title + stats_html + _table(["Sev","Table","Column","Issue Type","Count","%"], rows) + remeds
    btns = [] if tf else [
        (f"🛡️ {tbl.upper()} issues", f"DQ issues in {tbl.upper()} table")
        for tbl in sorted({i["tableId"] for i in MOCK_DQ_ISSUES})
    ]
    return html, btns


def _ans_dq_dashboard():
    comp  = [("DM",98.4),("AE",99.9),("VS",99.8),("LB",98.4),("ADSL",99.6)]
    trend = [("Jan",45),("Feb",38),("Mar",27),("Apr",19),("May",8),("Jun",6)]
    bars  = "".join(_bar(t, ((v-97)/3)*100, "#0d9488", f"{v}%") for t, v in comp)
    max_v = max(v for _, v in trend)
    trend_bars = "".join(
        f'<div style="display:flex;flex-direction:column;align-items:center;flex:1">'
        f'<div style="width:100%;background:#0d9488;border-radius:3px 3px 0 0;'
        f'min-height:3px;height:{int((v/max_v)*52)}px" title="{v} issues"></div>'
        f'<span style="font-size:0.67rem;color:#94a3b8;margin-top:3px">{m}</span>'
        f'<span style="font-size:0.67rem;font-weight:600;color:#475569">{v}</span>'
        f'</div>'
        for m, v in trend
    )
    html = (
        "<strong>📊 DQ Dashboard — STUDY-2026-FIBER</strong>"
        + _cards(
            ("Total records", "10,240", None),
            ("Active exceptions", "6", "#dc2626"),
            ("Avg completeness", "99.2%", "#0d9488"),
            ("Error reduction 6mo", "↓87%", "#16a34a"),
            ("Tables monitored", "5", None),
        )
        + _section("Data completeness by table")
        + bars
        + _section("Anomaly count — 2026")
        + f'<div style="display:flex;align-items:flex-end;gap:4px;height:72px;'
          f'padding:0 4px;margin-top:4px">{trend_bars}</div>'
        + _opt_note("↓87% error reduction since January 2026 — 3 critical blockers remain")
    )
    return html, [("🛡️ View full DQ audit", "Show all data quality issues and remediations")]


def _ans_er():
    rels = [
        ("DM","AE","1:Many","USUBJID + STUDYID","One subject → many adverse events"),
        ("DM","VS","1:Many","USUBJID + STUDYID","One subject → many vital sign timepoints"),
        ("DM","LB","1:Many","USUBJID + STUDYID","One subject → many lab results"),
        ("DM","ADSL","1:1","USUBJID","One subject → one analysis record (ADaM)"),
    ]
    nodes = (
        '<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;'
        'background:#f8fafc;border-radius:10px;padding:10px 14px;margin:8px 0">'
        '<span style="background:#0d9488;color:white;padding:5px 12px;border-radius:8px;'
        'font-weight:700;font-size:0.82rem">DM</span>'
        + "".join(
            f'<span style="color:#94a3b8;font-size:1rem">→</span>'
            f'<span style="border:1.5px solid #e2e8f0;background:#fff;padding:5px 12px;'
            f'border-radius:8px;font-size:0.82rem;font-weight:600">{to}</span>'
            for _, to, *_ in rels
        )
        + '<span style="margin-left:auto;font-size:0.72rem;color:#94a3b8">'
          'All linked via <code>USUBJID</code></span>'
        + '</div>'
    )
    rows = [
        [f'<strong>{r[0]}</strong>', f'<strong>{r[1]}</strong>',
         f'<code style="background:#e0f2fe;color:#0369a1;padding:1px 6px;border-radius:4px">{r[2]}</code>',
         f'<code style="font-size:0.7rem">{r[3]}</code>',
         r[4]]
        for r in rels
    ]
    html = (
        "<strong>🔗 ER Relationships — All tables linked via USUBJID</strong>"
        + nodes
        + _table(["Parent","Child","Cardinality","Join Key","Rule"], rows)
        + _opt_note(
            "<strong>DM</strong> is the anchor table. Every SDTM domain joins to DM via "
            "<code>USUBJID</code>. ADSL is the ADaM subject-level dataset derived from DM."
        )
    )
    return html, [
        (f"📋 {t['name']} schema", f"Describe the {t['name']} table columns")
        for t in CLINICAL_TABLES
    ]


def _ans_dictionary():
    rows = [
        [f'<strong>{tn}</strong>', _stag(std),
         f'<code style="color:#0f766e">{c["name"]}</code>',
         f'<code style="color:#64748b;font-size:0.68rem">{c["type"]}</code>',
         "✓" if not c["nullable"] else "○",
         c["description"][:65] + ("…" if len(c["description"])>65 else "")]
        for tn, std, c in _ALL_COLS
    ]
    html = (
        f'<strong>📖 Data Dictionary — {len(_ALL_COLS)} columns · {len(CLINICAL_TABLES)} tables</strong>'
        + _cards(
            ("Total columns", len(_ALL_COLS), None),
            ("Primary keys", sum(1 for _, _, c in _ALL_COLS if c["isPrimary"]), None),
            ("Foreign keys", sum(1 for _, _, c in _ALL_COLS if c.get("isForeign")), None),
            ("Required fields", sum(1 for _, _, c in _ALL_COLS if not c["nullable"]), None),
        )
        + _table(["Table","Std","Column","Type","Req","Description"], rows, max_h="300px")
        + _opt_note('Ask about any column: "What is ARMCD?" · "Explain LBNRIND" · "What does SAFFL mean?"')
    )
    return html, []


def _flag_chips(flags):
    parts = "&nbsp;&nbsp;".join(
        '<span style="background:#f0fdf4;border:1px solid #86efac;'
        'border-radius:12px;padding:1px 9px;font-size:0.72rem;'
        'color:#15803d;font-weight:600">' + fl + '</span>'
        for fl in flags
    )
    return f'<div style="margin-bottom:6px">{parts}</div>'


def _ans_column(tname: str, cname: str):
    c = _COL_MAP.get((tname, cname))
    if not c:
        return _ans_fallback()
    t = _TABLE_MAP[tname]
    flags = []
    if c["isPrimary"]: flags.append("🔑 Primary Key")
    if c.get("isForeign"):
        flags.append(f'🔗 FK → {c.get("foreignTable","")}.{c.get("foreignColumn","")}')
    samples = " &nbsp;|&nbsp; ".join(
        f'<code style="font-size:0.76rem">{s}</code>' for s in c["sampleData"][:4]
    )
    html = (
        f'<strong>📖 {tname}.<code>{cname}</code></strong> {_stag(t["standard"])}<br>'
        + _cards(
            ("Table", tname, None), ("Data type", c["type"], None),
            ("Required", "Yes" if not c["nullable"] else "No", None),
            ("CDISC standard", c["cdiscStandard"], None),
        )
        + f'<div style="margin:8px 0;font-size:0.84rem;color:#1e293b;line-height:1.6">'
          f'{c["description"]}</div>'
        + (_flag_chips(flags) if flags else "")
        + _section("Sample values")
        + f'<div style="margin-top:3px">{samples}</div>'
    )
    return html, [
        (f"📋 {tname} full schema", f"Describe the {tname} table columns"),
        ("📖 Full dictionary", "Show full data dictionary for all columns"),
    ]


def _ans_glossary_all():
    rows = [
        [f'<strong>{g["term"]}</strong>',
         f'<span style="background:#dbeafe;color:#1d4ed8;border-radius:20px;'
         f'padding:1px 7px;font-size:0.68rem;font-weight:700">{g["standard"]}</span>',
         g["category"],
         g["definition"][:80] + "…"]
        for g in GLOSSARY
    ]
    html = (
        f'<strong>📚 Clinical & GCP Glossary — {len(GLOSSARY)} terms</strong>'
        + _table(["Term","Standard","Category","Definition"], rows, max_h="280px")
        + _opt_note('Ask about any term: "What is MedDRA?" · "Explain ALCOA+" · "Define ITT"')
    )
    btns = [(f"📚 {g['term']}", f"What is {g['term']}?") for g in GLOSSARY[:8]]
    return html, btns


def _ans_glossary_term(term: str):
    g = _GLOS_MAP.get(term.lower())
    if not g:
        return _ans_glossary_all()
    cat_color = {
        "Terminology Standards": "#7c3aed", "Data Standards": "#0369a1",
        "Analysis Standards": "#0369a1", "Regulatory Compliance": "#dc2626",
        "Data Integrity": "#ca8a04", "Domain Identifiers": "#0f766e",
        "Analysis Populations": "#16a34a", "Safety Reporting": "#dc2626",
        "Evidence Generation": "#475569", "Analysis Variables": "#0369a1",
        "Adverse Event Variables": "#ca8a04",
    }.get(g["category"], "#475569")
    html = (
        f'<strong>📚 {g["term"]}</strong> '
        f'<span style="background:#dbeafe;color:#1d4ed8;border-radius:20px;'
        f'padding:1px 7px;font-size:0.7rem;font-weight:700">{g["standard"]}</span>'
        f'<span style="font-size:0.72rem;color:{cat_color};margin-left:6px;font-weight:600">'
        f'{g["category"]}</span><br><br>'
        f'<span style="font-size:0.88rem;line-height:1.65;color:#1e293b">{g["definition"]}</span>'
        f'<br><br>'
        f'<div style="background:#f8fafc;border-left:3px solid #0d9488;border-radius:0 8px 8px 0;'
        f'padding:8px 12px;font-size:0.8rem;color:#475569;margin-top:2px">'
        f'<strong style="color:#0f172a">Example:</strong> {g["example"]}</div>'
        + (f'<pre style="margin-top:8px;background:#f1f5f9;padding:5px 9px;border-radius:6px;'
           f'font-size:0.74rem;color:#334155">{g["code"]}</pre>' if g.get("code") else "")
    )
    related = [
        (f"📚 {og['term']}", f"What is {og['term']}?")
        for og in GLOSSARY
        if og["term"] != g["term"] and og["standard"] == g["standard"]
    ][:4]
    return html, [("📚 All glossary terms", "List all CDISC and GCP glossary terms")] + related


def _ans_regulatory():
    syn = [
        ["Study ID", "<strong>STUDY-2026-FIBER</strong>"],
        ["Title", "Phase III RCT — Drug X vs Placebo"],
        ["Sponsor", "DataGenome Pharma Inc."],
        ["Phase / Design", "III · Double-blind, randomized 1:1"],
        ["Indication", "Chronic Inflammatory Disease"],
        ["Primary Endpoint", "Reduction in disease severity score at Week 24"],
        ["Subjects", "250 randomized (Active n=125, Placebo n=125)"],
        ["CDISC Standard", "SDTM 3.3 + ADaM 1.3"],
        ["Data Cutoff", "May 31, 2026"],
        ["Submission Target", "FDA (NDA) + EMA (MAA)"],
    ]
    inv = [
        [f'<strong>{t["name"]}</strong>', t["label"], _stag(t["standard"]),
         f'{t["rowCount"]:,}',
         '<span style="background:#dcfce7;color:#15803d;border-radius:20px;'
         'padding:1px 8px;font-size:0.7rem;font-weight:700">✅ Validated</span>']
        for t in CLINICAL_TABLES
    ]
    blocking = [i for i in MOCK_DQ_ISSUES if i["severity"] == "Critical"]
    block_html = "".join(
        f'<div style="display:flex;align-items:flex-start;gap:6px;margin-top:4px;font-size:0.78rem">'
        f'<span style="background:#fee2e2;color:#dc2626;border-radius:20px;padding:1px 7px;'
        f'font-size:0.68rem;font-weight:700;flex-shrink:0;margin-top:1px">BLOCKER</span>'
        f'<span><strong>{i["tableId"].upper()}.{i["columnName"]}</strong> — '
        f'{i["issueType"]}: {i["description"]}</span></div>'
        for i in blocking
    )
    html = (
        "<strong>📜 Regulatory Dossier — STUDY-2026-FIBER</strong>"
        + _table(["Field", "Value"], syn)
        + _section("Dataset Inventory")
        + _table(["Domain","Label","Standard","Records","Status"], inv)
        + _section(f"Submission Blockers ({len(blocking)} Critical Issues)")
        + block_html
        + _opt_note(
            f'Total records: <strong>{sum(t["rowCount"] for t in CLINICAL_TABLES):,}</strong> · '
            f'<strong>{len(blocking)} critical issues must be resolved before database lock.</strong>'
        )
    )
    return html, [("🛡️ Full DQ audit", "Show all data quality issues and remediations")]


def _ans_ae():
    sevs = [("MILD",624,54.7,"#0d9488"),("MODERATE",398,34.9,"#eab308"),("SEVERE",118,10.4,"#ef4444")]
    soc = [
        ("Infections & infestations", 412, 36.1),
        ("Nervous system disorders", 287, 25.2),
        ("Gastrointestinal disorders", 195, 17.1),
        ("General disorders", 143, 12.5),
        ("Musculoskeletal disorders", 103, 9.0),
    ]
    sev_bars = "".join(_bar(l, p, c, f"{v} ({p}%)", 70) for l,v,p,c in sevs)
    soc_rows = [
        [s[0], str(s[1]), f'{s[2]}%',
         f'<div style="width:{int(s[2]*2)}px;height:8px;background:#0d9488;'
         f'border-radius:4px;display:inline-block"></div>']
        for s in soc
    ]
    html = (
        "<strong>⚠️ Adverse Events — AE Domain Summary</strong>"
        + _cards(
            ("Total AE records", "1,140", None),
            ("Serious (AESER=Y)", "47", "#dc2626"),
            ("SAE rate", "4.1%", "#dc2626"),
            ("Most common", "URTI", None),
            ("Subjects with AE", "198 / 250", None),
        )
        + _section("Severity breakdown")
        + sev_bars
        + _section("Top 5 System Organ Classes (MedDRA SOC)")
        + _table(["SOC", "Events", "%", ""], soc_rows)
        + _opt_note(
            "2 AE DQ issues active: future AESTDTC dates (Warning) "
            "and AEENDTC < AESTDTC constraint violations (Warning)"
        )
    )
    return html, [
        ("🛡️ AE DQ issues", "DQ issues in AE table"),
        ("📋 AE schema", "Describe the AE table columns"),
        ("📚 What is AESEV?", "What is AESEV?"),
        ("📚 What is MedDRA?", "What is MedDRA?"),
    ]


def _ans_populations():
    rows = [
        ["ITT", "Intent-to-Treat", "All randomized subjects",
         '<code>ITTFL = Y</code>', "Primary efficacy analysis",
         '<strong style="color:#0d9488">250</strong>'],
        ["SAF", "Safety Population", "≥1 dose received",
         '<code>SAFFL = Y</code>', "All AE analyses",
         '<strong style="color:#0d9488">247</strong>'],
        ["PP", "Per-Protocol", "No major deviations",
         "Protocol deviation flag", "Sensitivity analysis",
         '<strong style="color:#64748b">~218</strong>'],
    ]
    arm_rows = [
        ["ACT", "Active Treatment Drug X", "125", "Active arm",
         '<code>TRT01P / TRT01A</code> in ADSL'],
        ["PLAC", "Placebo Baseline Control", "125", "Control arm",
         '<code>TRT01P / TRT01A</code> in ADSL'],
    ]
    html = (
        "<strong>👥 Analysis Populations & Treatment Arms</strong>"
        + _section("Analysis populations (ADSL flags)")
        + _table(["Code","Name","Criteria","ADSL Flag","Use","n"], rows)
        + _section("Treatment arms (ARMCD / ACTARMCD in DM)")
        + _table(["Code","Label","n","Role","ADSL variable"], arm_rows)
        + _opt_note(
            "Age group <code>AGEGR1</code> in ADSL: <code>&lt;65</code> (n=189) · "
            "<code>≥65</code> (n=61). Use for stratified subgroup analyses."
        )
    )
    return html, [
        ("📚 What is ITT?", "What is ITT?"),
        ("📚 What is SAF?", "What is SAF?"),
        ("📋 ADSL schema", "Describe the ADSL table columns"),
    ]


def _ans_study():
    rows = [
        ["Study ID", "<strong>STUDY-2026-FIBER</strong>"],
        ["Title", "Phase III Double-blind RCT — Drug X vs Placebo"],
        ["Sponsor", "DataGenome Pharma Inc."],
        ["Phase", "III"],
        ["Indication", "Chronic Inflammatory Disease"],
        ["Design", "Randomized 1:1, double-blind, placebo-controlled"],
        ["Primary Endpoint", "Reduction in disease severity score at Week 24"],
        ["Key Secondary", "Safety profile, biomarker response, QoL"],
        ["Subjects", "250 randomized (ITT n=250, SAF n=247)"],
        ["Sites", "12 investigative sites"],
        ["CDISC", "SDTM 3.3 + ADaM 1.3"],
        ["Data Cutoff", "May 31, 2026"],
        ["Submission", "FDA (NDA) + EMA (MAA)"],
    ]
    html = (
        "<strong>🧬 Study Overview — STUDY-2026-FIBER</strong>"
        + _table(["Field", "Value"], rows)
        + _cards(
            ("Total records", "10,240", None),
            ("CDISC tables", "5", None),
            ("Active DQ issues", "6", "#dc2626"),
            ("Completeness", "99.2%", "#0d9488"),
        )
    )
    return html, [
        ("📋 All tables", "List all SDTM and ADaM tables"),
        ("📜 Reg dossier", "Show regulatory submission status and dossier"),
        ("🛡️ DQ status", "Show all data quality issues and remediations"),
    ]


def _ans_fallback():
    html = (
        "<span style='color:#64748b'>I can help with "
        "<code>STUDY-2026-FIBER</code>. Choose a topic:</span>"
    )
    btns = [
        ("📋 Schema Explorer",   "List all SDTM and ADaM tables"),
        ("🛡️ DQ Audit",          "Show all data quality issues and remediations"),
        ("📊 DQ Dashboard",      "Show DQ completeness metrics and error trends"),
        ("🔗 ER Relationships",  "Explain ER relationships between all tables"),
        ("📖 Data Dictionary",   "Show full data dictionary for all columns"),
        ("📚 GCP Glossary",      "List all CDISC and GCP glossary terms"),
        ("📜 Regulatory Report", "Show regulatory submission status and dossier"),
        ("⚠️ Adverse Events",   "Summarise AE domain data and severity breakdown"),
    ]
    return html, btns


# ─── Main dispatcher ──────────────────────────────────────────────────────────
def generate_answer(query: str, chunks: list) -> tuple:
    """
    Route query → correct answer builder.
    Returns (html_string, buttons_list).
    """
    intent = _detect_intent(query, chunks)

    if intent == "help":             return _ans_help()
    if intent == "schema_list":      return _ans_schema_list()
    if intent == "dq_audit":         return _ans_dq_audit(query)
    if intent == "dq_dashboard":     return _ans_dq_dashboard()
    if intent == "er":               return _ans_er()
    if intent == "dictionary":       return _ans_dictionary()
    if intent == "glossary_all":     return _ans_glossary_all()
    if intent == "regulatory":       return _ans_regulatory()
    if intent == "ae_summary":       return _ans_ae()
    if intent == "populations":      return _ans_populations()
    if intent == "study":            return _ans_study()

    if intent.startswith("table_schema:"):
        return _ans_table_schema(intent.split(":")[1])
    if intent.startswith("column:"):
        _, tname, cname = intent.split(":")
        return _ans_column(tname, cname)
    if intent.startswith("glossary_term:"):
        return _ans_glossary_term(intent.split(":", 1)[1])

    return _ans_fallback()
