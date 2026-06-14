"""
dynamic_answer.py — Answer engine for user-uploaded datasets
Handles all chatbot queries against live DatasetInfo objects.
Stats, schema, DQ, relationships, distributions — all from actual data.
"""
import re
import pandas as pd
from data_loader import DatasetInfo

# ── HTML helpers ──────────────────────────────────────────────────────────────
def _c(label, value, color=None):
    st = f'style="color:{color}"' if color else ""
    return (f'<div class="sc"><div class="sc-l">{label}</div>'
            f'<div class="sc-v" {st}>{value}</div></div>')

def _cards(*items):
    return '<div class="sc-row">' + "".join(_c(*i) for i in items) + "</div>"

def _bar(label, pct, color="#0d9488", note="", lw=160):
    return (
        f'<div class="bar-row">'
        f'<span style="width:{lw}px;font-size:0.76rem;color:#475569;'
        f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{label}</span>'
        f'<div class="bar-bg"><div class="bar-fill" '
        f'style="width:{min(float(pct),100):.1f}%;background:{color}"></div></div>'
        f'<span style="width:90px;text-align:right;color:#64748b;font-size:0.74rem">{note}</span>'
        f'</div>'
    )

def _tbl(headers, rows, max_h=None):
    scroll = f'style="max-height:{max_h};overflow-y:auto"' if max_h else ""
    ths = "".join(
        f'<th style="padding:6px 10px;text-align:left;font-size:0.7rem;'
        f'color:#64748b;border-bottom:1px solid #e2e8f0;background:#f8fafc;'
        f'white-space:nowrap">{h}</th>' for h in headers
    )
    trs = "".join(
        "<tr>" + "".join(
            f'<td style="padding:5px 10px;font-size:0.75rem;'
            f'border-bottom:1px solid #f1f5f9;vertical-align:top">{cell}</td>'
            for cell in row
        ) + "</tr>"
        for row in rows
    )
    return (f'<div style="border:1px solid #e2e8f0;border-radius:8px;'
            f'overflow:hidden;margin-top:6px;{scroll}">'
            f'<table style="width:100%;border-collapse:collapse">'
            f'<thead><tr>{ths}</tr></thead><tbody>{trs}</tbody></table></div>')

def _sec(t):
    return (f'<div style="margin:10px 0 4px;font-size:0.68rem;font-weight:700;'
            f'color:#94a3b8;text-transform:uppercase;letter-spacing:.06em">{t}</div>')

def _note(t, color="#94a3b8"):
    return f'<div style="font-size:0.73rem;color:{color};margin-top:5px">{t}</div>'

def _warn(t):
    return (f'<div style="background:#fff7ed;border:1px solid #fed7aa;'
            f'border-radius:6px;padding:5px 9px;font-size:0.76rem;'
            f'color:#c2410c;margin-top:5px">{t}</div>')

def _sev_tag(sev):
    cls = {"Critical":"#dc2626","Warning":"#ca8a04"}.get(sev,"#2563eb")
    bg  = {"Critical":"#fee2e2","Warning":"#fef9c3"}.get(sev,"#dbeafe")
    return (f'<span style="background:{bg};color:{cls};border-radius:20px;'
            f'padding:1px 7px;font-size:0.67rem;font-weight:700">{sev}</span>')

COLORS = ["#0d9488","#2563eb","#7c3aed","#ca8a04","#dc2626","#0369a1","#15803d","#c026d3"]

# ── Intent helpers ────────────────────────────────────────────────────────────
def _find_dataset(query: str, datasets: list):
    """Find which dataset the user is asking about."""
    ql = query.lower()
    for ds in datasets:
        if ds.name.lower() in ql:
            return ds
    return None

def _find_column(query: str, ds: DatasetInfo):
    """Find which column in a dataset."""
    ql = query.lower()
    # Exact match first
    for c in ds.columns:
        if c.name.lower() in ql:
            return c
    # Partial match
    for c in ds.columns:
        if any(part in ql for part in c.name.lower().split('_') if len(part) > 2):
            return c
    return None

def _find_column_any(query: str, datasets: list):
    """Find (dataset, column) from query across all datasets."""
    ql = query.lower()
    for ds in datasets:
        for c in ds.columns:
            if c.name.lower() in ql:
                return ds, c
    return None, None

STAT_WORDS = {
    "max": ["max","maximum","highest","largest","oldest","biggest","peak","upper","top"],
    "min": ["min","minimum","lowest","smallest","youngest","least","bottom","lower","floor"],
    "mean":   ["mean","average","avg"],
    "median": ["median","middle"],
    "std":    ["std","deviation","spread","variability"],
    "count":  ["count","how many","number","total","n =","size"],
    "missing":["missing","null","blank","empty","incomplete"],
    "unique": ["unique","distinct","different"],
    "distribution":["distribution","breakdown","split","frequency","freq",
                    "proportion","percentage","percent","composition","ratio"],
}

def _detect_stat(ql):
    for stat, kws in STAT_WORDS.items():
        if any(k in ql for k in kws):
            return stat
    return None

# ── Answer builders ───────────────────────────────────────────────────────────

def ans_overview(report: dict):
    s  = report["summary"]
    ds = report["datasets"]
    rows = [
        [f'<strong>{d.name}</strong>',
         d.source.upper(),
         f'{d.row_count:,}',
         str(d.col_count),
         d.description[:60]+"…" if len(d.description)>60 else d.description]
        for d in ds
    ]
    html = (
        "<strong>📊 Dataset Collection — Analysis Complete</strong>"
        + _cards(
            ("Datasets",       s["total_datasets"],     None),
            ("Total rows",     f'{s["total_rows"]:,}',  None),
            ("Total columns",  s["total_cols"],          None),
            ("Relationships",  s["total_relationships"], "#0d9488"),
            ("DQ issues",      s["total_dq_issues"],
             "#dc2626" if s["critical_dq"] > 0 else "#ca8a04"),
            ("Shared cols",    s["shared_col_groups"],   None),
        )
        + _sec("Loaded datasets")
        + _tbl(["Table/File","Source","Rows","Cols","Description"], rows)
    )
    btns = [(f"📋 {d.name} schema", f"Show schema of {d.name}") for d in ds]
    btns += [
        ("🔗 Relationships",  "Show all relationships between datasets"),
        ("🛡️ DQ Issues",     "Show all data quality issues"),
    ]
    return html, btns


def ans_schema(ds: DatasetInfo, report: dict):
    pk_cands = report["pk_candidates"].get(ds.name, [])
    rows = []
    for c in ds.columns:
        flags = "🔑" if c.name in pk_cands else ""
        stat = ""
        if c.is_numeric and c.min_val is not None:
            stat = f'min={c.min_val} · max={c.max_val} · mean={c.mean_val}'
        elif c.top_values:
            top = list(c.top_values.items())[:2]
            stat = " · ".join(f"{k}({v})" for k,v in top)
        miss_color = "#dc2626" if c.missing_pct > 10 else ("#ca8a04" if c.missing_pct > 0 else "#15803d")
        rows.append([
            f'<code style="color:#0f766e">{c.name}</code>',
            c.dtype,
            flags or "—",
            f'<span style="color:{miss_color}">{c.missing} ({c.missing_pct}%)</span>',
            str(c.unique_count),
            stat or f'sample: {", ".join(c.sample[:2])}',
        ])
    html = (
        f'<strong>📋 Schema — {ds.name}</strong> '
        f'<span style="background:#e0f2fe;color:#0369a1;border-radius:12px;'
        f'padding:1px 8px;font-size:0.7rem;font-weight:700">{ds.source.upper()}</span><br>'
        f'<span style="font-size:0.77rem;color:#64748b">{ds.description}</span>'
        + _cards(
            ("Rows",    f'{ds.row_count:,}', None),
            ("Columns", ds.col_count,        None),
            ("PK candidates", len(pk_cands), "#0d9488" if pk_cands else None),
            ("With missing",  sum(1 for c in ds.columns if c.missing > 0),
             "#dc2626" if any(c.missing > 0 for c in ds.columns) else "#15803d"),
        )
        + _tbl(["Column","Type","PK","Missing","Unique","Stats/Sample"], rows)
    )
    return html, [
        ("📊 Statistics", f"Show statistics for {ds.name}"),
        ("🛡️ DQ issues",  f"Show DQ issues in {ds.name}"),
        ("📊 Overview",   "Show dataset overview"),
    ]


def ans_stats_column(ds: DatasetInfo, col, stat: str, query: str):
    c = col
    ql = query.lower()

    if c.is_numeric:
        stat_map = {
            "max":    ("Maximum",       c.max_val,    "#dc2626"),
            "min":    ("Minimum",       c.min_val,    "#2563eb"),
            "mean":   ("Mean",          c.mean_val,   "#0d9488"),
            "median": ("Median",        c.median_val, "#7c3aed"),
            "std":    ("Std Deviation", c.std_val,    "#ca8a04"),
            "count":  ("Row count",     ds.row_count, None),
            "missing":("Missing",       c.missing,    "#dc2626" if c.missing > 0 else "#15803d"),
            "unique": ("Unique values", c.unique_count, None),
        }
        s = stat or "max"
        lbl, val, col_c = stat_map.get(s, ("Value","—",None))
        highlight = (
            f'<div style="background:#f0fdf4;border:1px solid #86efac;border-radius:8px;'
            f'padding:8px 12px;margin-bottom:8px">'
            f'<span style="color:#64748b">{lbl} of <code>{ds.name}.{c.name}</code>: </span>'
            f'<strong style="font-size:1.2rem;color:{col_c or "#0f172a"}">{val}</strong></div>'
        )
        html = (
            f'<strong>📊 {ds.name}.{c.name} — Statistics</strong><br>'
            + highlight
            + _cards(
                ("Maximum", c.max_val,    "#dc2626"),
                ("Minimum", c.min_val,    "#2563eb"),
                ("Mean",    c.mean_val,   "#0d9488"),
                ("Median",  c.median_val, "#7c3aed"),
                ("Std Dev", c.std_val,    "#ca8a04"),
                ("Count",   f'{ds.row_count:,}', None),
                ("Missing", f'{c.missing} ({c.missing_pct}%)',
                 "#dc2626" if c.missing > 0 else "#15803d"),
                ("Unique",  c.unique_count, None),
            )
        )
    elif c.top_values:
        return ans_distribution(ds, col)
    else:
        html = (
            f'<strong>📊 {ds.name}.{c.name}</strong><br>'
            + _cards(
                ("Count",   f'{ds.row_count:,}', None),
                ("Unique",  c.unique_count, None),
                ("Missing", f'{c.missing} ({c.missing_pct}%)',
                 "#dc2626" if c.missing > 0 else "#15803d"),
            )
            + _note(f'Sample values: {", ".join(c.sample[:5])}')
        )
    return html, [(f"📋 {ds.name} schema", f"Show schema of {ds.name}")]


def ans_distribution(ds: DatasetInfo, col):
    c = col
    if not c.top_values:
        return ans_stats_column(ds, col, "count", "")
    total  = ds.row_count
    bars   = ""
    for i, (k, v) in enumerate(sorted(c.top_values.items(), key=lambda x: -x[1])):
        pct = v / total * 100 if total > 0 else 0
        bars += _bar(str(k), pct, COLORS[i % len(COLORS)], f"{v:,} ({pct:.1f}%)")
    html = (
        f'<strong>📊 {ds.name}.{c.name} — Distribution</strong>'
        + _cards(
            ("Total rows",    f'{total:,}', None),
            ("Unique values", c.unique_count, None),
            ("Missing",       f'{c.missing} ({c.missing_pct}%)',
             "#dc2626" if c.missing > 0 else "#15803d"),
            ("Most common",   list(c.top_values.keys())[0] if c.top_values else "—", "#0d9488"),
        )
        + _sec(f"Value breakdown (top {len(c.top_values)} of {c.unique_count} unique)")
        + bars
    )
    return html, [(f"📋 {ds.name} schema", f"Show schema of {ds.name}")]


def ans_dq(report: dict, ds: DatasetInfo = None):
    issues = report["dq_issues"]
    if ds:
        issues = [i for i in issues if i["table"] == ds.name]
    if not issues:
        name   = ds.name if ds else "all datasets"
        ds_all = report["datasets"]
        rows_checked = sum(d.row_count for d in ([ds] if ds else ds_all))
        cols_checked = sum(d.col_count for d in ([ds] if ds else ds_all))
        return (
            f'<strong>✅ No DQ issues found in {name}</strong><br>'
            f'<div style="background:#f0fdf4;border:1px solid #86efac;border-radius:8px;'
            f'padding:8px 12px;margin-top:6px;font-size:.83rem;color:#15803d">'
            f'All {rows_checked:,} rows and {cols_checked} columns checked — '
            f'no missing values, negative anomalies, constant columns, or low-cardinality issues detected.'
            f'</div>',
            [("📊 Overview", "Show dataset overview")]
        )

    crit = [i for i in issues if i["severity"] == "Critical"]
    warn = [i for i in issues if i["severity"] == "Warning"]
    rows = [
        [_sev_tag(i["severity"]),
         f'<strong>{i["table"]}</strong>',
         f'<code>{i["column"]}</code>',
         i["type"],
         f'{i["count"]:,}',
         f'{i["pct"]}%',
         i["detail"][:60] + ("…" if len(i["detail"]) > 60 else "")]
        for i in issues
    ]
    title = f'<strong>🛡️ DQ Issues — {ds.name}</strong>' if ds \
            else f'<strong>🛡️ DQ Issues — All Datasets ({len(issues)} issues)</strong>'
    html  = (
        title
        + _cards(
            ("Critical", len(crit), "#dc2626"),
            ("Warnings",  len(warn), "#ca8a04"),
            ("Affected columns", len({i["column"] for i in issues}), None),
        )
        + _tbl(["Sev","Table","Column","Type","Count","%","Detail"], rows, max_h="280px")
    )
    return html, [("📊 Overview", "Show dataset overview")]


def ans_relationships(report: dict):
    rels = report["relationships"]
    if not rels:
        return (
            '<strong>🔗 Relationships</strong><br>'
            + _note("No column relationships detected across datasets. "
                    "Datasets may not share common key columns.", "#ca8a04"),
            [("📊 Overview", "Show dataset overview")]
        )
    high = [r for r in rels if r["confidence"] >= 70]
    rows = [
        [f'<strong>{r["from_table"]}</strong>',
         f'<code>{r["from_col"]}</code>',
         f'<strong>{r["to_table"]}</strong>',
         f'<code>{r["to_col"]}</code>',
         r["type"],
         f'<span style="color:{"#0d9488" if r["confidence"]>=70 else "#ca8a04"};font-weight:600">'
         f'{r["confidence"]}%</span>',
         f'{r["overlap_pct"]}%']
        for r in rels[:20]
    ]
    # ER visual
    node_names = list({r["from_table"] for r in rels} | {r["to_table"] for r in rels})
    nodes_html = "".join(
        f'<span style="background:{"#0d9488" if i==0 else "#e0f2fe"};'
        f'color:{"white" if i==0 else "#0369a1"};'
        f'padding:4px 10px;border-radius:8px;font-size:0.8rem;font-weight:600">{n}</span>'
        for i, n in enumerate(node_names[:8])
    )
    html = (
        "<strong>🔗 Detected Relationships</strong>"
        + _cards(
            ("Total links",     len(rels), None),
            ("High confidence", len(high), "#0d9488"),
            ("Tables involved", len(node_names), None),
        )
        + _sec("Table map")
        + f'<div style="display:flex;gap:6px;flex-wrap:wrap;background:#f8fafc;'
          f'border-radius:8px;padding:8px 12px;margin:6px 0">{nodes_html}</div>'
        + _sec("Relationship details")
        + _tbl(["From Table","From Col","To Table","To Col","Type","Confidence","Overlap"], rows)
        + _note("Confidence = name similarity + value overlap. ≥70% = high confidence join candidate.")
    )
    return html, [
        ("📊 Overview",  "Show dataset overview"),
        ("🛡️ DQ Issues", "Show all data quality issues"),
    ]


def ans_missing(report: dict, ds: DatasetInfo = None):
    datasets = [ds] if ds else report["datasets"]
    rows = []
    for d in datasets:
        for c in d.columns:
            if c.missing > 0:
                sev_color = "#dc2626" if c.missing_pct > 20 else "#ca8a04"
                rows.append([
                    f'<strong>{d.name}</strong>',
                    f'<code>{c.name}</code>',
                    f'<span style="color:{sev_color}">{c.missing:,}</span>',
                    f'{c.missing_pct}%',
                    f'{d.row_count:,}',
                    "Critical" if c.missing_pct > 20 else "Warning",
                ])
    if not rows:
        return '<strong>✅ No missing values detected</strong>', []
    html = (
        "<strong>🛡️ Missing Values Report</strong>"
        + _tbl(["Table","Column","Missing","% Missing","Total Rows","Severity"], rows, max_h="280px")
    )
    return html, [("🛡️ Full DQ audit", "Show all data quality issues")]


def ans_full_stats(ds: DatasetInfo):
    sections = []
    for c in ds.columns:
        if c.is_numeric and c.min_val is not None:
            sections.append(
                f'<div style="background:#f8fafc;border:1px solid #e2e8f0;'
                f'border-radius:7px;padding:7px 10px;margin-top:5px">'
                f'<div style="font-size:0.76rem;font-weight:700;color:#0f172a;margin-bottom:4px">'
                f'<code>{c.name}</code> <span style="color:#94a3b8;font-weight:400">numeric</span></div>'
                + _cards(
                    ("Max",     c.max_val,    "#dc2626"),
                    ("Min",     c.min_val,    "#2563eb"),
                    ("Mean",    c.mean_val,   "#0d9488"),
                    ("Median",  c.median_val, "#7c3aed"),
                    ("Std Dev", c.std_val,    "#ca8a04"),
                    ("Missing", f'{c.missing} ({c.missing_pct}%)',
                     "#dc2626" if c.missing > 0 else "#15803d"),
                )
                + '</div>'
            )
        elif c.top_values:
            freq_bars = "".join(
                _bar(str(k), v / ds.row_count * 100, COLORS[i % len(COLORS)],
                     f"{v:,} ({v/ds.row_count*100:.1f}%)")
                for i, (k, v) in enumerate(list(c.top_values.items())[:5])
            )
            sections.append(
                f'<div style="background:#f8fafc;border:1px solid #e2e8f0;'
                f'border-radius:7px;padding:7px 10px;margin-top:5px">'
                f'<div style="font-size:0.76rem;font-weight:700;color:#0f172a;margin-bottom:4px">'
                f'<code>{c.name}</code> <span style="color:#94a3b8;font-weight:400">categorical · '
                f'{c.unique_count} unique</span></div>'
                + freq_bars + '</div>'
            )
    html = (
        f'<strong>📊 Full Statistics — {ds.name}</strong><br>'
        f'<span style="font-size:0.77rem;color:#64748b">{ds.description}</span>'
        + "".join(sections)
    )
    return html, [(f"📋 {ds.name} schema", f"Show schema of {ds.name}")]


def ans_shared_cols(report: dict):
    shared = report["shared_cols"]
    if not shared:
        return '<strong>No shared columns found across datasets.</strong>', []
    rows = [
        [k.upper(),
         " · ".join(f'<code>{t}.{c}</code>' for t,c in v),
         str(len(v))]
        for k, v in sorted(shared.items(), key=lambda x: -len(x[1]))
    ]
    html = (
        f'<strong>🔗 Shared Columns — {len(shared)} common fields</strong>'
        + _tbl(["Normalized name","Appears in","Tables"], rows)
        + _note("Shared columns are candidates for JOIN keys between datasets.")
    )
    return html, [("🔗 Relationships", "Show all relationships between datasets")]


# ── Main dispatcher ───────────────────────────────────────────────────────────
def generate_dynamic_answer(query: str, chunks: list, report: dict) -> tuple:
    """Route natural language query to the right answer builder."""
    ql       = query.lower()
    datasets = report["datasets"]
    ds       = _find_dataset(query, datasets)
    stat     = _detect_stat(ql)

    # Overview / summary
    if any(w in ql for w in ["overview","summary","all dataset","collection",
                               "what dataset","what data","what table","list dataset",
                               "how many dataset","what have","what is loaded"]):
        return ans_overview(report)

    # Relationships / ER
    if any(w in ql for w in ["relationship","relation","join","link","connect",
                               "foreign key","er diagram","common col","shared col"]):
        if any(w in ql for w in ["shared","common"]):
            return ans_shared_cols(report)
        return ans_relationships(report)

    # DQ issues
    if any(w in ql for w in ["dq","quality","issue","problem","error","missing",
                               "null","duplicate","anomaly","violation","invalid"]):
        return ans_dq(report, ds)

    # Schema / structure
    if any(w in ql for w in ["schema","structure","column","field","variable",
                               "describe","show me","what col"]):
        if ds:
            return ans_schema(ds, report)
        if datasets:
            return ans_schema(datasets[0], report)

    # Statistics — column-level
    dsi, col = _find_column_any(query, datasets)
    if col:
        if any(w in ql for w in ["distribution","breakdown","freq","how many",
                                   "count","split","proportion"]):
            return ans_distribution(dsi, col)
        return ans_stats_column(dsi, col, stat, query)

    # Statistics — table-level
    if stat in ("count","missing","unique") and ds:
        if stat == "missing":
            return ans_missing(report, ds)
        return ans_full_stats(ds)

    if any(w in ql for w in ["statistic","stat","profile","numeric","analytic"]) and ds:
        return ans_full_stats(ds)

    if ds:
        return ans_schema(ds, report)

    # Fallback — show overview
    return ans_overview(report)
