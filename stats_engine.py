"""
stats_engine.py — Natural language → dataset statistics
Handles queries like:
  "Give the maximum age in the DM table"
  "What is the mean ALT in LB?"
  "How many subjects are male?"
  "Count of MILD adverse events"
  "Distribution of RACE in DM"
  "Missing values in LB"
"""
import re
from data import DATASET_STATS, CLINICAL_TABLES

# ── Column/table aliases ──────────────────────────────────────────────────────
_TABLE_ALIASES = {
    "dm": "DM", "demographics": "DM", "demographic": "DM",
    "ae": "AE", "adverse event": "AE", "adverse events": "AE",
    "vs": "VS", "vital sign": "VS", "vital signs": "VS", "vitals": "VS",
    "lb": "LB", "lab": "LB", "labs": "LB", "laboratory": "LB",
    "adsl": "ADSL", "analysis": "ADSL",
}
_COL_ALIASES = {
    "age": ("DM", "AGE"), "ages": ("DM", "AGE"),
    "sex": ("DM", "SEX"), "gender": ("DM", "SEX"),
    "male": ("DM", "SEX"), "female": ("DM", "SEX"), "subjects": ("DM", "AGE"),
    "race": ("DM", "RACE"), "ethnicity": ("DM", "RACE"),
    "armcd": ("DM", "ARMCD"), "arm": ("DM", "ARMCD"), "treatment arm": ("DM", "ARMCD"),
    "sysbp": ("VS", "SYSBP"), "systolic": ("VS", "SYSBP"), "systolic bp": ("VS", "SYSBP"),
    "diabp": ("VS", "DIABP"), "diastolic": ("VS", "DIABP"), "diastolic bp": ("VS", "DIABP"),
    "pulse": ("VS", "PULSE"), "heart rate": ("VS", "PULSE"),
    "temp": ("VS", "TEMP"), "temperature": ("VS", "TEMP"),
    "alt": ("LB", "ALT"), "alanine": ("LB", "ALT"),
    "ast": ("LB", "AST"), "aspartate": ("LB", "AST"),
    "wbc": ("LB", "WBC"), "white blood": ("LB", "WBC"),
    "hgb": ("LB", "HGB"), "hemoglobin": ("LB", "HGB"), "haemoglobin": ("LB", "HGB"),
    "creat": ("LB", "CREAT"), "creatinine": ("LB", "CREAT"),
    "plat": ("LB", "PLAT"), "platelet": ("LB", "PLAT"), "platelets": ("LB", "PLAT"),
    "aeseq": ("AE", "AESEQ"),
    "aesev": ("AE", "AESEV"), "severity": ("AE", "AESEV"),
    "aeser": ("AE", "AESER"), "serious": ("AE", "AESER"),
    "agegr1": ("ADSL", "AGEGR1"), "age group": ("ADSL", "AGEGR1"),
    "saffl": ("ADSL", "SAFFL"), "safety flag": ("ADSL", "SAFFL"),
    "ittfl": ("ADSL", "ITTFL"), "itt flag": ("ADSL", "ITTFL"),
    "trt01p": ("ADSL", "TRT01P"), "treatment": ("ADSL", "TRT01P"),
}
_STAT_KEYWORDS = {
    "max": ["maximum", "max", "highest", "largest", "oldest", "biggest", "peak", "upper"],
    "min": ["minimum", "min", "lowest", "smallest", "youngest", "least", "bottom", "lower"],
    "mean": ["mean", "average", "avg", "typical", "central"],
    "median": ["median", "middle", "50th percentile", "50th"],
    "std": ["std", "standard deviation", "deviation", "spread", "variability"],
    "count": ["count", "how many", "number of", "total", "n =", "sample size", "size"],
    "missing": ["missing", "null", "blank", "empty", "incomplete", "na ", "n/a"],
    "unique": ["unique", "distinct", "different values", "cardinality", "how many different"],
    "distribution": ["distribution", "breakdown", "split", "frequency", "freq", "proportion",
                     "percentage", "percent", "ratio", "composition"],
}

def _detect_stat(ql: str) -> str | None:
    for stat, keywords in _STAT_KEYWORDS.items():
        if any(kw in ql for kw in keywords):
            return stat
    return None

def _detect_table_col(ql: str):
    """Returns (table, col) or (table, None) or (None, None)"""
    # Try column aliases first (more specific)
    for alias, (tbl, col) in _COL_ALIASES.items():
        if alias in ql:
            return tbl, col
    # Try table aliases
    for alias, tbl in _TABLE_ALIASES.items():
        if alias in ql:
            return tbl, None
    return None, None

def is_stats_query(query: str) -> bool:
    """Returns True if this query is asking for dataset statistics."""
    ql = query.lower()
    has_stat    = _detect_stat(ql) is not None
    tbl, col    = _detect_table_col(ql)
    has_target  = tbl is not None or col is not None
    # Also catch natural phrasing without explicit stat word
    natural = any(p in ql for p in [
        "give me", "tell me", "what is the", "show me the",
        "how many", "find the", "get the", "list the"
    ])
    return (has_stat and has_target) or (natural and has_target and has_stat)

# ── HTML helpers (reuse from answer_engine style) ────────────────────────────
def _card(label, value, color=None):
    style = f'style="color:{color}"' if color else ""
    return (f'<div class="sc"><div class="sc-l">{label}</div>'
            f'<div class="sc-v" {style}>{value}</div></div>')

def _cards(*items):
    return '<div class="sc-row">' + "".join(_card(*i) for i in items) + "</div>"

def _bar(label, pct, color="#0d9488", note=""):
    return (
        f'<div class="bar-row">'
        f'<span style="width:180px;font-size:0.76rem;color:#475569">{label}</span>'
        f'<div class="bar-bg"><div class="bar-fill" style="width:{min(pct,100):.1f}%;background:{color}"></div></div>'
        f'<span style="width:70px;text-align:right;color:#64748b;font-size:0.74rem">{note}</span>'
        f'</div>'
    )

def _section(t):
    return (f'<div style="margin:10px 0 4px;font-size:0.68rem;font-weight:700;'
            f'color:#94a3b8;text-transform:uppercase;letter-spacing:.06em">{t}</div>')

def _dq_note(note):
    return (f'<div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:6px;'
            f'padding:5px 9px;font-size:0.76rem;color:#c2410c;margin-top:6px">{note}</div>')

# ── Stat fetchers ─────────────────────────────────────────────────────────────
def _get_numeric_stats(tbl, col_key):
    """Pull numeric stats dict. VS and LB are nested by test code."""
    s = DATASET_STATS.get(tbl, {})
    if tbl == "VS":
        return s.get("VSSTRESN", {}).get(col_key)
    if tbl == "LB":
        return s.get("LBSTRESN", {}).get(col_key)
    return s.get(col_key)

def _get_freq_stats(tbl, col_key):
    return DATASET_STATS.get(tbl, {}).get(col_key)

# ── Answer builders ───────────────────────────────────────────────────────────
def _ans_numeric(stat, tbl, col, data, query):
    ql = query.lower()
    unit = data.get("unit", "")
    note = data.get("note")

    # Single-stat answer (max/min/mean/median/std)
    stat_map = {
        "max":    ("Maximum", data.get("max"), "#dc2626"),
        "min":    ("Minimum", data.get("min"), "#2563eb"),
        "mean":   ("Mean (Average)", round(data.get("mean", 0), 2), "#0d9488"),
        "median": ("Median", data.get("median"), "#7c3aed"),
        "std":    ("Std Deviation", round(data.get("std", 0), 2), "#ca8a04"),
        "count":  ("Record count", data.get("count"), None),
        "missing":("Missing values", data.get("missing", 0),
                   "#dc2626" if data.get("missing", 0) > 0 else "#15803d"),
        "unique": ("Distinct values", data.get("unique", "N/A"), None),
    }

    # Determine which stats to highlight
    primary_stat = _detect_stat(ql)
    label, value, color = stat_map.get(primary_stat, ("Value", "—", None))

    # Summary card block
    cards_html = _cards(
        ("Maximum", f'{data["max"]} {unit}'.strip(), "#dc2626"),
        ("Minimum", f'{data["min"]} {unit}'.strip(), "#2563eb"),
        ("Mean",    f'{round(data["mean"],2)} {unit}'.strip(), "#0d9488"),
        ("Median",  f'{data["median"]} {unit}'.strip(), "#7c3aed"),
        ("Std Dev", f'{round(data["std"],2)} {unit}'.strip(), "#ca8a04"),
        ("Count",   f'{data["count"]:,}', None),
        ("Missing", str(data.get("missing", 0)),
         "#dc2626" if data.get("missing", 0) > 0 else "#15803d"),
    )

    # Highlight the asked stat prominently
    highlight = (
        f'<div style="background:#f0fdf4;border:1px solid #86efac;border-radius:8px;'
        f'padding:8px 12px;margin-bottom:8px;font-size:0.88rem">'
        f'<span style="color:#64748b">{label} of <code>{tbl}.{col}</code>:</span> '
        f'<strong style="font-size:1.15rem;color:{color or "#0f172a"}">'
        f'{value} {unit}</strong></div>'
    )

    html = (
        f'<strong>📊 {tbl}.{col} — Descriptive Statistics</strong><br>'
        + highlight
        + cards_html
        + (f'<br>{_dq_note(note)}' if note else "")
    )
    return html, [
        (f"📋 {tbl} schema", f"Describe the {tbl} table columns"),
        ("🛡️ DQ issues", "Show all data quality issues and remediations"),
    ]


def _ans_distribution(tbl, col, data, query):
    freq = data.get("freq", {})
    total = data.get("count", sum(freq.values()))
    missing = data.get("missing", 0)
    note = data.get("note")

    colors = ["#0d9488","#2563eb","#7c3aed","#ca8a04","#dc2626","#0369a1","#15803d"]
    bars = ""
    for i, (k, v) in enumerate(sorted(freq.items(), key=lambda x: -x[1])):
        pct = (v / total * 100) if total > 0 else 0
        bars += _bar(str(k), pct, colors[i % len(colors)], f"{v:,} ({pct:.1f}%)")

    cards_html = _cards(
        ("Total records", f'{total:,}', None),
        ("Unique values", str(data.get("unique", len(freq))), None),
        ("Missing", str(missing), "#dc2626" if missing > 0 else "#15803d"),
        ("Most common", max(freq, key=freq.get) if freq else "—", "#0d9488"),
    )

    html = (
        f'<strong>📊 {tbl}.{col} — Frequency Distribution</strong><br>'
        + cards_html
        + _section(f"Value breakdown (n={total:,})")
        + bars
        + (f'<br>{_dq_note(note)}' if note else "")
    )
    return html, [
        (f"📋 {tbl} schema", f"Describe the {tbl} table columns"),
    ]


def _ans_missing(tbl, all_cols_stats, query):
    rows_html = ""
    found_any = False
    for col, data in all_cols_stats.items():
        if isinstance(data, dict) and "missing" in data:
            missing = data.get("missing", 0)
            count   = data.get("count", 0)
            pct     = (missing / count * 100) if count > 0 else 0
            color   = "#dc2626" if missing > 0 else "#15803d"
            status  = "⚠️ Has missing" if missing > 0 else "✅ Complete"
            rows_html += (
                f'<tr>'
                f'<td style="padding:5px 10px;font-size:0.75rem;border-bottom:1px solid #f1f5f9">'
                f'<code style="color:#0f766e">{col}</code></td>'
                f'<td style="padding:5px 10px;font-size:0.75rem;border-bottom:1px solid #f1f5f9;color:{color}">'
                f'{missing:,}</td>'
                f'<td style="padding:5px 10px;font-size:0.75rem;border-bottom:1px solid #f1f5f9">'
                f'{count:,}</td>'
                f'<td style="padding:5px 10px;font-size:0.75rem;border-bottom:1px solid #f1f5f9;color:{color}">'
                f'{pct:.1f}%</td>'
                f'<td style="padding:5px 10px;font-size:0.75rem;border-bottom:1px solid #f1f5f9">'
                f'{status}</td>'
                f'</tr>'
            )
            found_any = True

    if not found_any:
        return f'<strong>📊 {tbl} — Missing Value Report</strong><br>No missing value statistics available for {tbl}.', []

    html = (
        f'<strong>📊 {tbl} — Missing Value Report</strong>'
        f'<div style="border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;margin-top:6px">'
        f'<table style="width:100%;border-collapse:collapse">'
        f'<thead><tr>'
        f'<th style="padding:6px 10px;text-align:left;font-size:0.7rem;color:#64748b;border-bottom:1px solid #e2e8f0;background:#f8fafc">Column</th>'
        f'<th style="padding:6px 10px;text-align:left;font-size:0.7rem;color:#64748b;border-bottom:1px solid #e2e8f0;background:#f8fafc">Missing</th>'
        f'<th style="padding:6px 10px;text-align:left;font-size:0.7rem;color:#64748b;border-bottom:1px solid #e2e8f0;background:#f8fafc">Total</th>'
        f'<th style="padding:6px 10px;text-align:left;font-size:0.7rem;color:#64748b;border-bottom:1px solid #e2e8f0;background:#f8fafc">% Missing</th>'
        f'<th style="padding:6px 10px;text-align:left;font-size:0.7rem;color:#64748b;border-bottom:1px solid #e2e8f0;background:#f8fafc">Status</th>'
        f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
    )
    return html, [("🛡️ DQ issues", "Show all data quality issues and remediations")]


def _ans_table_summary(tbl, tbl_stats, query):
    """Full statistical profile of a table."""
    sections = []
    for col, data in tbl_stats.items():
        if not isinstance(data, dict):
            continue
        if "mean" in data:
            # Numeric
            unit = data.get("unit","")
            sections.append(
                f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:7px;'
                f'padding:7px 10px;margin-top:6px">'
                f'<div style="font-size:0.76rem;font-weight:700;color:#0f172a;margin-bottom:4px">'
                f'<code>{col}</code> <span style="color:#94a3b8;font-weight:400">numeric · {unit}</span></div>'
                + _cards(
                    ("Max",  f'{data["max"]} {unit}'.strip(), "#dc2626"),
                    ("Min",  f'{data["min"]} {unit}'.strip(), "#2563eb"),
                    ("Mean", f'{round(data["mean"],2)} {unit}'.strip(), "#0d9488"),
                    ("Missing", str(data.get("missing",0)),
                     "#dc2626" if data.get("missing",0) > 0 else "#15803d"),
                )
                + (f'<div style="font-size:0.72rem;color:#c2410c;margin-top:3px">{data["note"]}</div>'
                   if data.get("note") else "")
                + '</div>'
            )
        elif "freq" in data:
            # Categorical
            freq = data["freq"]
            top = max(freq, key=freq.get) if freq else "—"
            sections.append(
                f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:7px;'
                f'padding:7px 10px;margin-top:6px">'
                f'<div style="font-size:0.76rem;font-weight:700;color:#0f172a;margin-bottom:4px">'
                f'<code>{col}</code> <span style="color:#94a3b8;font-weight:400">categorical</span></div>'
                + _cards(
                    ("Count",   f'{data["count"]:,}', None),
                    ("Unique",  str(data.get("unique",len(freq))), None),
                    ("Missing", str(data.get("missing",0)),
                     "#dc2626" if data.get("missing",0)>0 else "#15803d"),
                    ("Top value", top, "#0d9488"),
                )
                + '</div>'
            )

    t = next((t for t in CLINICAL_TABLES if t["name"] == tbl), {})
    html = (
        f'<strong>📊 {tbl} — Full Statistical Profile</strong>'
        f'<br><span style="font-size:0.77rem;color:#64748b">'
        f'{t.get("description","")}</span>'
        + "".join(sections)
    )
    return html, [
        (f"📋 {tbl} schema",  f"Describe the {tbl} table columns"),
        ("🛡️ DQ issues",     "Show all data quality issues and remediations"),
    ]


# ── Main entry point ──────────────────────────────────────────────────────────
def answer_stats_query(query: str):
    """
    Parse natural language stats query and return (html, buttons).
    Returns None if query doesn't match any stats pattern.
    """
    ql = query.lower()
    stat   = _detect_stat(ql)
    tbl, col = _detect_table_col(ql)

    if tbl is None and col is None:
        return None  # not a stats query

    # VS/LB: col refers to test code, look up numeric data
    if tbl in ("VS", "LB") and col is not None:
        data = _get_numeric_stats(tbl, col)
        if data:
            if stat in ("distribution", None) and "freq" in data:
                return _ans_distribution(tbl, col, data, query)
            return _ans_numeric(stat, tbl, col, data, query)

    # DM/AE/ADSL with specific column
    if col is not None:
        data = _get_freq_stats(tbl, col)
        if data:
            if "freq" in data:
                if stat in ("distribution", "count", None) or any(
                    w in ql for w in ["how many","breakdown","split","frequency"]
                ):
                    return _ans_distribution(tbl, col, data, query)
            if "mean" in data:
                return _ans_numeric(stat, tbl, col, data, query)

    # Missing values query for a table
    if stat == "missing" and tbl:
        tbl_stats = DATASET_STATS.get(tbl, {})
        flat = {}
        for k, v in tbl_stats.items():
            if isinstance(v, dict) and "missing" in v:
                flat[k] = v
            elif isinstance(v, dict):  # nested (VS/LB)
                for kk, vv in v.items():
                    if isinstance(vv, dict) and "missing" in vv:
                        flat[kk] = vv
        return _ans_missing(tbl, flat, query)

    # Whole-table summary
    if tbl and stat in ("count", None) and col is None:
        tbl_stats = DATASET_STATS.get(tbl, {})
        if tbl_stats:
            flat = {}
            for k, v in tbl_stats.items():
                if isinstance(v, dict):
                    if "mean" in v or "freq" in v:
                        flat[k] = v
                    else:  # nested
                        for kk, vv in v.items():
                            if isinstance(vv, dict):
                                flat[kk] = vv
            return _ans_table_summary(tbl, flat, query)

    return None  # fallback
