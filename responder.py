"""
responder.py — NL intent router + rich HTML answer generator
All 8 abstract capabilities handled purely through chat NL.
No buttons for specific features — everything through query intent detection.
"""
import re, io, base64
import pandas as pd
from core import (
    render_er_svg, _smart_charts, generate_pdf, dict_to_csv,
    LS_GLOSSARY, ColProfile, TableProfile, retrieve
)

# ── HTML primitives ───────────────────────────────────────────────────────────
def _c(label, value, color=None):
    st = f'style="color:{color}"' if color else ""
    return (f'<div class="sc"><div class="sc-l">{label}</div>'
            f'<div class="sc-v" {st}>{value}</div></div>')
def _cards(*items):
    return '<div class="sc-row">'+"".join(_c(*i) for i in items)+"</div>"
def _bar(label, pct, color="#0d9488", note="", lw=160):
    pct = min(float(pct), 100)
    return (f'<div class="bar-row">'
            f'<span style="width:{lw}px;font-size:.76rem;color:#475569;'
            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{label}</span>'
            f'<div class="bar-bg"><div class="bar-fill" style="width:{pct:.1f}%;background:{color}"></div></div>'
            f'<span style="width:80px;text-align:right;color:#64748b;font-size:.74rem">{note}</span>'
            f'</div>')
def _tbl(headers, rows, max_h=None):
    scroll = f'style="max-height:{max_h};overflow-y:auto"' if max_h else ""
    ths = "".join(f'<th style="padding:5px 9px;text-align:left;font-size:.69rem;'
                  f'color:#64748b;border-bottom:1px solid #e2e8f0;background:#f8fafc;'
                  f'white-space:nowrap">{h}</th>' for h in headers)
    trs = "".join(
        "<tr>"+"".join(f'<td style="padding:4px 9px;font-size:.74rem;'
                       f'border-bottom:1px solid #f1f5f9;vertical-align:top">{cell}</td>'
                       for cell in row)+"</tr>"
        for row in rows
    )
    return (f'<div style="border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;margin-top:6px;{scroll}">'
            f'<table style="width:100%;border-collapse:collapse">'
            f'<thead><tr>{ths}</tr></thead><tbody>{trs}</tbody></table></div>')
def _sec(t):
    return (f'<div style="margin:10px 0 4px;font-size:.67rem;font-weight:700;'
            f'color:#94a3b8;text-transform:uppercase;letter-spacing:.06em">{t}</div>')
def _note(t, c="#94a3b8"):
    return f'<div style="font-size:.73rem;color:{c};margin-top:5px">{t}</div>'
def _warn(t):
    return (f'<div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:6px;'
            f'padding:5px 9px;font-size:.76rem;color:#c2410c;margin-top:5px">{t}</div>')
def _ok(t):
    return (f'<div style="background:#f0fdf4;border:1px solid #86efac;border-radius:6px;'
            f'padding:5px 9px;font-size:.76rem;color:#15803d;margin-top:5px">{t}</div>')
def _sev(s):
    cls={"Critical":"#dc2626","Warning":"#ca8a04"}.get(s,"#2563eb")
    bg ={"Critical":"#fee2e2","Warning":"#fef9c3"}.get(s,"#dbeafe")
    return (f'<span style="background:{bg};color:{cls};border-radius:20px;'
            f'padding:1px 7px;font-size:.67rem;font-weight:700">{s}</span>')
CLRS=["#0d9488","#2563eb","#7c3aed","#ca8a04","#dc2626","#0369a1","#15803d","#c026d3"]

# ── Intent detection ──────────────────────────────────────────────────────────
INTENTS = {
    "er":          ["er diagram","er map","entity relationship","erd","relationship diagram",
                    "table map","show relationship","draw er","generate er","visualize er"],
    "dashboard":   ["dashboard","chart","plot","visuali","graph","analytic","trend","distribution",
                    "show me a","create chart","create dashboard","build dashboard"],
    "pdf":         ["pdf","report","generate report","create report","export report","download report",
                    "regulatory report","compile report","create pdf","export pdf"],
    "dictionary":  ["dictionary","data dict","export dict","download dict","column definitions",
                    "all columns","column metadata","generate dictionary","create dictionary"],
    "dq":          ["data quality","dq","audit","quality check","missing","null","anomaly",
                    "violation","error","issue","problem","duplicate","negative value"],
    "schema":      ["schema","structure","describe","explain table","what columns","column list",
                    "table structure","fields in","columns in","show table","what is in"],
    "glossary":    ["glossary","what is","what does","define","meaning","term","terminology",
                    "explain what","business glossary","clinical term"],
    "relationship":["relationship","join","link","foreign key","connect","related","how are",
                    "shared column","common field","which tables"],
    "stats":       ["maximum","minimum","mean","average","median","std","count","how many",
                    "total","sum","statistics","profile","distribution of","breakdown","frequency"],
    "overview":    ["overview","summary","what data","what tables","what dataset","loaded",
                    "collection","all tables","list tables","show all"],
}

def detect_intent(query: str) -> str:
    ql = query.lower()
    for intent, kws in INTENTS.items():
        if any(k in ql for k in kws):
            return intent
    return "general"

def find_table(query: str, tables: list) -> TableProfile | None:
    ql = query.lower()
    for t in tables:
        if t.name.lower() in ql:
            return t
    return None

def find_col(query: str, tables: list):
    ql = query.lower()
    for t in tables:
        for c in t.columns:
            if c.name.lower() in ql and len(c.name) > 2:
                return t, c
    return None, None

def detect_stat_kw(ql: str) -> str:
    for stat, kws in [
        ("max",  ["maximum","max","highest","largest","oldest","biggest","peak"]),
        ("min",  ["minimum","min","lowest","smallest","youngest","least"]),
        ("mean", ["mean","average","avg"]),
        ("median",["median","middle"]),
        ("std",  ["std","deviation","spread"]),
        ("count",["count","how many","total","number of"]),
        ("missing",["missing","null","blank","incomplete"]),
        ("distribution",["distribution","breakdown","frequency","freq","proportion"]),
    ]:
        if any(k in ql for k in kws):
            return stat
    return "summary"

# ── Answer builders ───────────────────────────────────────────────────────────

def ans_overview(report):
    s  = report["summary"]
    ts = report["tables"]
    rows = []
    for t in ts:
        miss = sum(1 for c in t.columns if c.missing>0)
        pks  = ", ".join(t.pk_candidates[:2]) or "—"
        rows.append([
            f'<strong>{t.name}</strong>',
            t.source.upper(),
            f'{t.row_count:,}',
            str(t.col_count),
            str(miss),
            pks,
            t.description[:55]+"…" if len(t.description)>55 else t.description,
        ])
    html = (
        "<strong>🧬 DataGenome AI — Dataset Collection Overview</strong>"
        + _cards(
            ("Datasets",     s['n_tables'],             None),
            ("Total rows",   f'{s["total_rows"]:,}',   None),
            ("Total columns",s['total_cols'],            None),
            ("Relationships",s['n_rels'],               "#0d9488"),
            ("DQ issues",    s['n_dq'],
             "#dc2626" if s['n_crit']>0 else "#ca8a04"),
        )
        + _sec("Loaded datasets")
        + _tbl(["Table","Source","Rows","Cols","Cols w/Missing","PK","Description"], rows)
        + _note("Ask: 'explain patients table' · 'create dashboard' · 'show ER diagram' · 'generate PDF' · 'show DQ issues'")
    )
    return html

def ans_schema(t: TableProfile, report):
    rows = []
    for c in t.columns:
        pk    = "🔑" if c.name in t.pk_candidates else ""
        dtype = "📊 numeric" if c.is_numeric else ("📅 date" if c.is_date else "📝 text")
        stat  = ""
        if c.is_numeric and c.min_val is not None:
            stat = f"min={c.min_val} · max={c.max_val} · mean={c.mean_val}"
        elif c.top_values:
            stat = " · ".join(f"{k}({v})" for k,v in list(c.top_values.items())[:3])
        miss_c = "#dc2626" if c.missing_pct>10 else ("#ca8a04" if c.missing_pct>0 else "#15803d")
        rows.append([
            f'<code style="color:#0f766e">{c.name}</code>',
            dtype, pk,
            f'<span style="color:{miss_c}">{c.missing}({c.missing_pct}%)</span>',
            str(c.unique_count),
            stat or f'sample: {", ".join(c.sample[:3])}',
            c.ai_description[:55]+"…" if len(c.ai_description)>55 else c.ai_description,
        ])
    html = (
        f'<strong>📋 Schema Intelligence — {t.name}</strong> '
        f'<span style="background:#e0f2fe;color:#0369a1;border-radius:12px;'
        f'padding:1px 8px;font-size:.69rem;font-weight:700">{t.source.upper()}</span><br>'
        f'<span style="font-size:.77rem;color:#64748b">{t.description}</span>'
        + _cards(
            ("Rows",    f'{t.row_count:,}',      None),
            ("Columns", t.col_count,              None),
            ("PK candidates", len(t.pk_candidates), "#0d9488" if t.pk_candidates else None),
            ("Cols w/missing", sum(1 for c in t.columns if c.missing>0),
             "#dc2626" if any(c.missing>0 for c in t.columns) else "#15803d"),
            ("Numeric cols", sum(1 for c in t.columns if c.is_numeric), None),
        )
        + _tbl(["Column","Type","PK","Missing","Unique","Stats/Sample","AI Description"], rows)
    )
    return html

def ans_er(report):
    svg = render_er_svg(report)
    rels = report["relationships"]
    rel_rows = [
        [f'<strong>{r["from_table"]}</strong>',
         f'<code>{r["from_col"]}</code>',
         f'<strong>{r["to_table"]}</strong>',
         f'<code>{r["to_col"]}</code>',
         r["type"],
         f'<span style="color:{"#0d9488" if r["confidence"]>=70 else "#ca8a04"};font-weight:600">{r["confidence"]}%</span>',
         f'{r["overlap_pct"]}%']
        for r in rels[:12]
    ]
    html = (
        "<strong>🔗 Entity-Relationship Diagram</strong>"
        + _cards(
            ("Tables",        len(report["tables"]),                None),
            ("Relationships", len(rels),                            "#0d9488"),
            ("High confidence",len([r for r in rels if r["confidence"]>=70]),"#0d9488"),
        )
        + f'<div style="margin:10px 0">{svg}</div>'
        + (
            _sec("Relationship detail table")
            + _tbl(["From Table","From Col","To Table","To Col","Type","Confidence","Value Overlap"], rel_rows)
            + _note("Confidence = name similarity × 55% + value overlap × 45%. ≥70% = high-confidence join candidate.")
            if rels else _warn("No relationships detected. Datasets may not share common key columns.")
        )
    )
    return html

def ans_dashboard(report, query):
    chart_html = _smart_charts(report, query)
    s = report["summary"]
    html = (
        "<strong>📊 Analytics Dashboard</strong>"
        + _cards(
            ("Datasets",   s['n_tables'],          None),
            ("Total rows", f'{s["total_rows"]:,}', None),
            ("Charts",     "Auto-generated",        "#0d9488"),
        )
        + f'<div style="margin-top:10px">{chart_html}</div>'
        + _note("Ask for specific charts: 'age distribution', 'response score by treatment arm', 'lab results trend'")
    )
    return html

def ans_dq(report, tbl=None):
    issues = [i for i in report["dq_issues"] if not tbl or i["table"]==tbl.name] if tbl else report["dq_issues"]
    crit   = [i for i in issues if i["severity"]=="Critical"]
    warns  = [i for i in issues if i["severity"]=="Warning"]
    if not issues:
        checked = f"{tbl.row_count:,} rows" if tbl else f"{report['summary']['total_rows']:,} rows"
        return (
            f'<strong>✅ No DQ Issues Found'
            f'{" — " + tbl.name if tbl else ""}</strong>'
            + _ok(f"All {checked} and {tbl.col_count if tbl else report['summary']['total_cols']} columns checked. "
                  f"No missing values, negatives, constants, or duplicate PKs detected.")
        )
    rows = [
        [_sev(i["severity"]),
         f'<strong>{i["table"]}</strong>',
         f'<code>{i["column"]}</code>',
         i["type"],
         f'{i["count"]:,}',
         f'{i["pct"]}%',
         i["detail"],
         ]
        for i in issues
    ]
    remed = "".join(
        f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:7px;'
        f'padding:7px 10px;margin-top:5px;font-size:.76rem">'
        f'{_sev(i["severity"])} <strong>{i["table"]}.{i["column"]}</strong> — {i["type"]}<br>'
        f'<span style="color:#475569">{i["detail"]}</span><br>'
        f'<span style="color:#0d9488;font-weight:600">▶ Recommended fix: </span>'
        f'<span style="color:#0d9488">{i["fix"]}</span></div>'
        for i in issues
    )
    title = f'DQ Audit — {tbl.name}' if tbl else f'Data Quality Audit — {len(issues)} Issues'
    html = (
        f'<strong>🛡️ {title}</strong>'
        + _cards(
            ("Critical",  len(crit),  "#dc2626"),
            ("Warnings",  len(warns), "#ca8a04"),
            ("Tables affected", len({i["table"] for i in issues}), None),
            ("Total affected rows", sum(i["count"] for i in issues), None),
        )
        + _tbl(["Severity","Table","Column","Issue Type","Count","%","Detail"], rows, max_h="280px")
        + _sec("Remediation guidance")
        + remed
    )
    return html

def ans_dictionary(report):
    rows = []
    for t in report["tables"]:
        for c in t.columns:
            dtype = "numeric" if c.is_numeric else ("date" if c.is_date else "text")
            pk    = "🔑" if c.name in t.pk_candidates else ""
            rows.append([
                f'<strong>{t.name}</strong>',
                f'<code style="color:#0f766e">{c.name}</code>',
                dtype, pk,
                f'<span style="color:{"#dc2626" if c.missing_pct>10 else "#15803d"}">{c.missing}({c.missing_pct}%)</span>',
                str(c.unique_count),
                c.ai_description[:70]+"…" if len(c.ai_description)>70 else c.ai_description,
            ])
    total = sum(t.col_count for t in report["tables"])
    html = (
        f'<strong>📖 AI Data Dictionary — {total} columns</strong>'
        + _cards(
            ("Tables",       len(report["tables"]),                   None),
            ("Total columns",total,                                    None),
            ("Numeric cols", sum(1 for t in report["tables"] for c in t.columns if c.is_numeric), None),
            ("Date cols",    sum(1 for t in report["tables"] for c in t.columns if c.is_date),    None),
        )
        + _tbl(["Table","Column","Type","PK","Missing","Unique","AI Description"], rows, max_h="320px")
        + _note("💡 Type 'export data dictionary' to download as CSV")
    )
    return html

def ans_glossary(query: str, report=None):
    ql = query.lower()
    # Single term lookup
    match = next((g for g in LS_GLOSSARY if g.lower() in ql and len(g)>2), None)
    if match:
        defn = LS_GLOSSARY[match]
        html = (
            f'<strong>📚 {match}</strong>'
            f'<div style="background:#f0fdf4;border-left:3px solid #0d9488;border-radius:0 8px 8px 0;'
            f'padding:8px 12px;margin:8px 0;font-size:.85rem;color:#1e293b;line-height:1.65">'
            f'{defn}</div>'
        )
        # Related terms
        rel = {k:v for k,v in LS_GLOSSARY.items()
               if k!=match and any(w in v.lower() for w in match.lower().split()[:2])}
        if rel:
            html += _sec("Related terms")
            for k,v in list(rel.items())[:4]:
                html += f'<div style="font-size:.78rem;margin:3px 0"><strong>{k}</strong>: {v}</div>'
        return html

    # All glossary — filter to loaded cols
    all_col_names = set()
    if report:
        all_col_names = {c.name.upper() for t in report["tables"] for c in t.columns}
    rows = []
    for term, defn in LS_GLOSSARY.items():
        tag = "✓" if term.upper() in all_col_names else ""
        rows.append([f'<strong>{term}</strong>', tag,
                     defn[:80]+"…" if len(defn)>80 else defn])
    html = (
        f'<strong>📚 Life Sciences Business Glossary — {len(LS_GLOSSARY)} terms</strong>'
        + _tbl(["Term","In Data","Definition"], rows, max_h="300px")
        + _note("Ask: 'what is ALT?' · 'define SDTM' · 'what does GCP mean?'")
    )
    return html

def ans_relationship(report):
    rels = report["relationships"]
    shared = report.get("shared_cols",{})
    if not rels:
        return (
            '<strong>🔗 Table Relationships</strong><br>'
            + _warn("No column relationships detected. Tables may not share common keys.")
        )
    rows = [
        [f'<strong>{r["from_table"]}</strong>', f'<code>{r["from_col"]}</code>',
         f'<strong>{r["to_table"]}</strong>',   f'<code>{r["to_col"]}</code>',
         r["type"],
         f'<span style="font-weight:600;color:{"#0d9488" if r["confidence"]>=70 else "#ca8a04"}">{r["confidence"]}%</span>',
         f'{r["overlap_pct"]}%']
        for r in rels
    ]
    # Visual node map
    node_names = list({r["from_table"] for r in rels}|{r["to_table"] for r in rels})
    nodes_html = "".join(
        f'<span style="background:{"#0d9488" if i==0 else "#e0f2fe"};'
        f'color:{"white" if i==0 else "#0369a1"};padding:4px 12px;border-radius:8px;'
        f'font-size:.8rem;font-weight:600">{n}</span> '
        for i,n in enumerate(node_names[:8])
    )
    html = (
        '<strong>🔗 Table Relationships & Schema Linkage</strong>'
        + _cards(
            ("Relationships", len(rels), None),
            ("High confidence", len([r for r in rels if r["confidence"]>=70]), "#0d9488"),
            ("Shared column groups", len(shared), None),
        )
        + _sec("Table map")
        + f'<div style="display:flex;flex-wrap:wrap;gap:6px;background:#f8fafc;'
          f'border-radius:8px;padding:8px 12px;margin:6px 0">{nodes_html}</div>'
        + _sec("Relationship detail")
        + _tbl(["From Table","From Col","To Table","To Col","Cardinality","Confidence","Value Overlap"], rows)
        + _note("Type 'show ER diagram' to render a visual entity-relationship map.")
    )
    return html

def ans_stats_col(t: TableProfile, c: ColProfile, stat: str, query: str):
    ql = query.lower()
    if c.is_numeric:
        stat_map = {
            "max":    ("Maximum",       c.max_val,    "#dc2626"),
            "min":    ("Minimum",       c.min_val,    "#2563eb"),
            "mean":   ("Mean",          c.mean_val,   "#0d9488"),
            "median": ("Median",        c.median_val, "#7c3aed"),
            "std":    ("Std Deviation", c.std_val,    "#ca8a04"),
            "count":  ("Row Count",     t.row_count,  None),
            "missing":("Missing",       c.missing,    "#dc2626" if c.missing>0 else "#15803d"),
        }
        lbl, val, clr = stat_map.get(stat, ("Value","—",None))
        highlight = (
            f'<div style="background:#f0fdf4;border:1px solid #86efac;border-radius:8px;'
            f'padding:8px 12px;margin-bottom:8px">'
            f'<span style="color:#64748b">{lbl} of <code>{t.name}.{c.name}</code>: </span>'
            f'<strong style="font-size:1.2rem;color:{clr or "#0f172a"}">{val}</strong></div>'
        )
        html = (
            f'<strong>📊 {t.name}.{c.name} — Statistics</strong><br>'
            + highlight
            + _cards(
                ("Maximum", c.max_val, "#dc2626"),
                ("Minimum", c.min_val, "#2563eb"),
                ("Mean",    c.mean_val,"#0d9488"),
                ("Median",  c.median_val,"#7c3aed"),
                ("Std Dev", c.std_val, "#ca8a04"),
                ("Unique",  c.unique_count, None),
                ("Missing", f'{c.missing}({c.missing_pct}%)',
                 "#dc2626" if c.missing>0 else "#15803d"),
            )
        )
    elif c.top_values:
        total  = t.row_count
        bars   = ""
        for i,(k,v) in enumerate(sorted(c.top_values.items(), key=lambda x:-x[1])[:10]):
            pct = v/total*100 if total>0 else 0
            bars += _bar(str(k), pct, CLRS[i%len(CLRS)], f"{v:,} ({pct:.1f}%)")
        html = (
            f'<strong>📊 {t.name}.{c.name} — Distribution</strong>'
            + _cards(
                ("Total rows",    f'{total:,}', None),
                ("Unique values", c.unique_count, None),
                ("Missing",       f'{c.missing}({c.missing_pct}%)',
                 "#dc2626" if c.missing>0 else "#15803d"),
                ("Most common",   list(c.top_values.keys())[0] if c.top_values else "—", "#0d9488"),
            )
            + _sec(f"Value breakdown")
            + bars
        )
    else:
        html = (
            f'<strong>📊 {t.name}.{c.name}</strong>'
            + _cards(
                ("Rows",    f'{t.row_count:,}', None),
                ("Unique",  c.unique_count, None),
                ("Missing", f'{c.missing}({c.missing_pct}%)',
                 "#dc2626" if c.missing>0 else "#15803d"),
            )
            + _note(f'Sample: {", ".join(c.sample[:5])}')
        )
    return html

def ans_full_stats(t: TableProfile):
    sections = []
    for c in t.columns:
        if c.is_numeric and c.min_val is not None:
            sections.append(
                f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:7px;'
                f'padding:7px 10px;margin-top:5px">'
                f'<div style="font-size:.76rem;font-weight:700;color:#0f172a;margin-bottom:4px">'
                f'<code>{c.name}</code> <span style="color:#94a3b8;font-weight:400">numeric</span></div>'
                + _cards(
                    ("Max",    c.max_val,    "#dc2626"),
                    ("Min",    c.min_val,    "#2563eb"),
                    ("Mean",   c.mean_val,   "#0d9488"),
                    ("Median", c.median_val, "#7c3aed"),
                    ("Std",    c.std_val,    "#ca8a04"),
                    ("Missing",f'{c.missing}({c.missing_pct}%)',
                     "#dc2626" if c.missing>0 else "#15803d"),
                ) + '</div>'
            )
        elif c.top_values:
            freq_bars = "".join(
                _bar(str(k),v/t.row_count*100,CLRS[i%len(CLRS)],
                     f"{v:,}({v/t.row_count*100:.1f}%)")
                for i,(k,v) in enumerate(list(c.top_values.items())[:6])
            )
            sections.append(
                f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:7px;'
                f'padding:7px 10px;margin-top:5px">'
                f'<div style="font-size:.76rem;font-weight:700;color:#0f172a;margin-bottom:4px">'
                f'<code>{c.name}</code> <span style="color:#94a3b8;font-weight:400">'
                f'categorical · {c.unique_count} unique</span></div>'
                + freq_bars + '</div>'
            )
    return (
        f'<strong>📊 Full Statistical Profile — {t.name}</strong><br>'
        f'<span style="font-size:.77rem;color:#64748b">{t.description}</span>'
        + "".join(sections)
    )

def ans_pdf_ready(report):
    return (
        '<strong>📄 PDF Report Ready</strong>'
        + _ok(
            "✅ Regulatory-style PDF generated covering: Schema Intelligence, "
            "ER Relationships, DQ Audit, AI Data Dictionary, and Business Glossary. "
            "Click the download button below to save."
        )
        + _note("The PDF report includes all loaded datasets and follows regulatory documentation standards.")
    )

def ans_dict_ready():
    return (
        '<strong>📖 Data Dictionary Export Ready</strong>'
        + _ok(
            "✅ Full column-level AI Data Dictionary generated as CSV. "
            "Includes table, column, data type, PK flag, missing stats, and AI descriptions. "
            "Click the download button below."
        )
    )

# ── Main router ───────────────────────────────────────────────────────────────
def respond(query: str, report: dict, rag_idx, rag_chunks) -> dict:
    """
    Returns dict: {html, action, action_data}
    action: None | "pdf" | "dict_csv" | "dashboard"
    """
    ql      = query.lower()
    intent  = detect_intent(query)
    tables  = report["tables"]
    tbl     = find_table(query, tables)
    t_obj, c_obj = find_col(query, tables)

    # ── ER Diagram ────────────────────────────────────────────────────────────
    if intent == "er":
        return {"html": ans_er(report), "action": None}

    # ── Dashboard ─────────────────────────────────────────────────────────────
    if intent == "dashboard":
        return {"html": ans_dashboard(report, query), "action": None}

    # ── PDF Report ────────────────────────────────────────────────────────────
    if intent == "pdf":
        pdf_bytes = generate_pdf(report)
        return {"html": ans_pdf_ready(report), "action":"pdf", "action_data": pdf_bytes}

    # ── Data Dictionary export ────────────────────────────────────────────────
    if intent == "dictionary" and any(k in ql for k in ["export","download","csv","generate","create"]):
        csv_str = dict_to_csv(report)
        return {"html": ans_dict_ready(), "action":"dict_csv", "action_data": csv_str.encode()}

    # ── Data Dictionary view ──────────────────────────────────────────────────
    if intent == "dictionary":
        return {"html": ans_dictionary(report), "action": None}

    # ── DQ Audit ──────────────────────────────────────────────────────────────
    if intent == "dq":
        return {"html": ans_dq(report, tbl), "action": None}

    # ── Schema / table explanation ────────────────────────────────────────────
    if intent == "schema":
        t = tbl or (tables[0] if tables else None)
        if t:
            return {"html": ans_schema(t, report), "action": None}

    # ── Glossary ──────────────────────────────────────────────────────────────
    if intent == "glossary":
        return {"html": ans_glossary(query, report), "action": None}

    # ── Relationships ─────────────────────────────────────────────────────────
    if intent == "relationship":
        return {"html": ans_relationship(report), "action": None}

    # ── Overview ──────────────────────────────────────────────────────────────
    if intent == "overview":
        return {"html": ans_overview(report), "action": None}

    # ── Stats — column-level ──────────────────────────────────────────────────
    if intent == "stats" and c_obj:
        stat = detect_stat_kw(ql)
        return {"html": ans_stats_col(t_obj, c_obj, stat, query), "action": None}

    # ── Stats — table-level ───────────────────────────────────────────────────
    if intent == "stats" and tbl:
        return {"html": ans_full_stats(tbl), "action": None}

    # ── RAG fallback ──────────────────────────────────────────────────────────
    hits = retrieve(rag_idx, rag_chunks, query, top_k=5)
    if hits:
        top = hits[0]
        cat = top.get("cat","")
        if cat == "dq":
            return {"html": ans_dq(report, tbl), "action": None}
        if cat == "schema":
            t = top["data"] if isinstance(top["data"], TableProfile) else tbl
            if t: return {"html": ans_schema(t, report), "action": None}
        if cat == "column" and isinstance(top["data"], tuple):
            tn, col = top["data"]
            t_match = next((t for t in tables if t.name==tn), None)
            if t_match:
                return {"html": ans_stats_col(t_match, col, detect_stat_kw(ql), query), "action": None}
        if cat == "relationship":
            return {"html": ans_relationship(report), "action": None}
        if cat == "glossary":
            return {"html": ans_glossary(query, report), "action": None}

    # ── Fallback: overview ────────────────────────────────────────────────────
    if tables:
        return {"html": ans_overview(report), "action": None}

    return {"html": '<span style="color:#94a3b8">Please upload a dataset first.</span>', "action": None}
