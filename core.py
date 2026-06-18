"""
core.py — DataGenome AI Core Engine
Handles: data loading, profiling, relationship detection, DQ audit,
ER diagram (SVG), dashboard (Plotly), PDF report, data dictionary,
business glossary, and full NL query routing.
"""
import re, io, json, csv, sqlite3, base64, warnings
import pandas as pd
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
warnings.filterwarnings("ignore")

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

# ─────────────────────────── Data Model ──────────────────────────────────────
@dataclass
class ColProfile:
    name: str
    dtype: str
    is_numeric: bool
    is_date: bool
    unique_count: int
    missing: int
    missing_pct: float
    sample: list
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean_val: Optional[float] = None
    median_val: Optional[float] = None
    std_val: Optional[float] = None
    top_values: dict = field(default_factory=dict)
    ai_description: str = ""

@dataclass
class TableProfile:
    name: str
    source: str
    row_count: int
    col_count: int
    columns: list          # list[ColProfile]
    df: object             # pd.DataFrame
    pk_candidates: list = field(default_factory=list)
    description: str = ""

# ─────────────────────────── Life sciences glossary ──────────────────────────
LS_GLOSSARY = {
    "PATIENT_ID":  "Unique subject identifier across the trial",
    "SITE_ID":     "Clinical site where the patient is enrolled",
    "AGE":         "Patient age in years at screening",
    "SEX":         "Biological sex (M/F)",
    "RACE":        "Self-reported racial category",
    "BMI":         "Body Mass Index (kg/m²)",
    "SMOKER":      "Smoking status at baseline (Y/N)",
    "DIABETES":    "Diabetes diagnosis at baseline (Y/N)",
    "TREATMENT_ARM": "Randomized treatment group (DRUG_X / PLACEBO)",
    "VISIT_WEEK":  "Study visit timepoint in weeks",
    "RESPONSE_SCORE": "Primary efficacy endpoint score (0–100)",
    "BLOOD_PRESSURE_SYS": "Systolic blood pressure (mmHg)",
    "BLOOD_PRESSURE_DIA": "Diastolic blood pressure (mmHg)",
    "CHOLESTEROL": "Total cholesterol (mg/dL)",
    "ADVERSE_EVENT": "Occurrence of any adverse event (Y/N)",
    "AE_SEVERITY": "Adverse event severity grade (MILD/MODERATE/SEVERE)",
    "ENROLLMENT_DATE": "Date of patient enrollment into the study",
    "COUNTRY":     "Country of the investigative site",
    "LAB_ID":      "Unique laboratory result identifier",
    "TEST_NAME":   "Full name of the laboratory test",
    "TEST_CODE":   "Short code for the laboratory test (CDISC LBTESTCD)",
    "RESULT_VALUE":"Numeric laboratory result value",
    "UNIT":        "Unit of measurement for the result",
    "NORMAL_LOW":  "Lower bound of the reference range",
    "NORMAL_HIGH": "Upper bound of the reference range",
    "ABNORMAL_FLAG":"Flag indicating result status (NORMAL/HIGH/LOW/CRITICAL)",
    "COLLECTION_DATE":"Date the sample was collected",
    "LAB_CATEGORY":"Category of lab panel (CHEMISTRY/HEMATOLOGY/URINALYSIS)",
    "ALT":         "Alanine Aminotransferase — liver enzyme, marker of hepatocellular damage",
    "AST":         "Aspartate Aminotransferase — liver/heart enzyme",
    "WBC":         "White Blood Cell Count — immune system marker",
    "HGB":         "Hemoglobin — oxygen-carrying protein in red blood cells",
    "CREAT":       "Creatinine — kidney function marker",
    "SDTM":        "Study Data Tabulation Model — FDA/PMDA submission standard",
    "ADaM":        "Analysis Data Model — statistical analysis standard",
    "MedDRA":      "Medical Dictionary for Regulatory Activities",
    "GCP":         "Good Clinical Practice — ICH E6(R2) ethical/scientific standard",
    "ALCOA+":      "Attributable, Legible, Contemporaneous, Original, Accurate + Complete",
    "ITT":         "Intent-to-Treat — all randomized subjects regardless of compliance",
    "SAF":         "Safety Population — subjects who received ≥1 dose",
    "DQ":          "Data Quality — completeness, accuracy, consistency of clinical data",
    "AE":          "Adverse Event — any untoward medical occurrence in a clinical trial",
    "SAE":         "Serious Adverse Event — AE resulting in death, hospitalization, etc.",
    "RWE":         "Real-World Evidence — evidence from non-trial healthcare settings",
    "ER Diagram":  "Entity-Relationship diagram showing table linkages and cardinalities",
    "FK":          "Foreign Key — column referencing a primary key in another table",
    "PK":          "Primary Key — unique row identifier within a table",
}

# ─────────────────────────── Profiler ────────────────────────────────────────
def _ai_describe(col_name: str, profile: ColProfile, table_name: str) -> str:
    n = col_name.upper()
    if n in LS_GLOSSARY:
        return LS_GLOSSARY[n]
    if profile.is_numeric:
        return (f"Numeric field in {table_name}. "
                f"Range: {profile.min_val}–{profile.max_val}, "
                f"mean {profile.mean_val}.")
    if profile.unique_count <= 5 and profile.top_values:
        vals = list(profile.top_values.keys())
        return f"Categorical field. Values: {', '.join(str(v) for v in vals[:6])}."
    return f"Field in {table_name} with {profile.unique_count} unique values."

def profile_df(df: pd.DataFrame, name: str, source: str) -> TableProfile:
    cols = []
    for col in df.columns:
        s       = df[col]
        missing = int(s.isna().sum())
        mpct    = round(missing / max(len(s), 1) * 100, 2)
        uniq    = int(s.nunique())
        sample  = [str(v) for v in s.dropna().head(4).tolist()]

        # Numeric
        num = pd.to_numeric(s, errors="coerce")
        is_num = num.notna().sum() > len(s) * 0.5

        # Date
        is_date = False
        if not is_num:
            try:
                pd.to_datetime(s.dropna().head(5), errors="raise")
                is_date = True
            except Exception:
                pass

        top_vals = {}
        if not is_num and uniq <= 300:
            vc = s.value_counts().head(10)
            top_vals = {str(k): int(v) for k, v in vc.items()}

        cp = ColProfile(
            name=str(col), dtype=str(s.dtype),
            is_numeric=is_num, is_date=is_date,
            unique_count=uniq, missing=missing, missing_pct=mpct,
            sample=sample, top_values=top_vals,
        )
        if is_num:
            cp.min_val    = round(float(num.min()), 4)
            cp.max_val    = round(float(num.max()), 4)
            cp.mean_val   = round(float(num.mean()), 4)
            cp.median_val = round(float(num.median()), 4)
            cp.std_val    = round(float(num.std()), 4)
        cols.append(cp)

    pks = [c.name for c in cols
           if c.unique_count == len(df) and c.missing == 0]

    tp = TableProfile(
        name=name, source=source,
        row_count=len(df), col_count=len(cols),
        columns=cols, df=df, pk_candidates=pks,
        description=(f"{len(df):,} rows × {len(cols)} columns | "
                     f"source: {source}"),
    )
    # AI descriptions
    for c in tp.columns:
        c.ai_description = _ai_describe(c.name, c, name)
    return tp

def load_file(file_obj) -> list:
    name = file_obj.name.lower()
    data = file_obj.read()
    results = []
    if name.endswith((".csv", ".tsv", ".txt")):
        sample = data[:4096].decode("utf-8", errors="replace")
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t|;")
            delim = dialect.delimiter
        except Exception:
            delim = ","
        df = pd.read_csv(io.BytesIO(data), delimiter=delim,
                         low_memory=False, on_bad_lines="skip")
        base = re.sub(r"\.(csv|tsv|txt)$", "", file_obj.name, flags=re.I)
        results.append(profile_df(df, base, "csv"))
    elif name.endswith((".xlsx", ".xls")):
        xl = pd.ExcelFile(io.BytesIO(data))
        base = re.sub(r"\.(xlsx|xls)$", "", file_obj.name, flags=re.I)
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            tname = f"{base}.{sheet}" if len(xl.sheet_names) > 1 else base
            results.append(profile_df(df, tname, "excel"))
    elif name.endswith(".json"):
        d = json.loads(data.decode("utf-8", errors="replace"))
        df = pd.json_normalize(d) if isinstance(d, list) else pd.DataFrame([d])
        base = re.sub(r"\.json$", "", file_obj.name, flags=re.I)
        results.append(profile_df(df, base, "json"))
    elif name.endswith((".db", ".sqlite", ".sqlite3")):
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            f.write(data); tmp = f.name
        try:
            conn = sqlite3.connect(tmp)
            tables = pd.read_sql(
                "SELECT name FROM sqlite_master WHERE type='table'", conn)
            for tname in tables["name"].tolist():
                df = pd.read_sql(f'SELECT * FROM "{tname}"', conn)
                results.append(profile_df(df, tname, "sqlite"))
            conn.close()
        finally:
            os.unlink(tmp)
    else:
        # Try CSV fallback
        df = pd.read_csv(io.BytesIO(data), low_memory=False, on_bad_lines="skip")
        results.append(profile_df(df, file_obj.name, "file"))
    return results

# ─────────────────────────── Relationship detection ──────────────────────────
def _norm(s): return re.sub(r'[^a-z0-9]', '', s.lower())

def detect_relationships(tables: list) -> list:
    rels = []
    seen = set()
    for i, ta in enumerate(tables):
        for j, tb in enumerate(tables):
            if i >= j: continue
            for ca in ta.columns:
                for cb in tb.columns:
                    na, nb = _norm(ca.name), _norm(cb.name)
                    sim = 1.0 if na == nb else (0.85 if (na in nb or nb in na) else 0.0)
                    if sim < 0.85: continue
                    key = tuple(sorted([(ta.name, ca.name),(tb.name, cb.name)]))
                    if key in seen: continue
                    seen.add(key)
                    # Value overlap
                    try:
                        va = set(ta.df[ca.name].dropna().astype(str).unique())
                        vb = set(tb.df[cb.name].dropna().astype(str).unique())
                        overlap = round(len(va & vb) / max(min(len(va),len(vb)),1)*100, 1)
                    except Exception:
                        overlap = 0.0
                    conf = round(sim*0.55 + (overlap/100)*0.45, 3)
                    rt = ("1:1" if ca.unique_count==ta.row_count and cb.unique_count==tb.row_count
                          else "1:Many" if ca.unique_count < cb.unique_count else "Many:1")
                    rels.append(dict(
                        from_table=ta.name, from_col=ca.name,
                        to_table=tb.name,   to_col=cb.name,
                        type=rt, confidence=round(conf*100,1),
                        overlap_pct=overlap,
                    ))
    return sorted(rels, key=lambda r: -r["confidence"])

def detect_dq(tables: list) -> list:
    issues = []
    for t in tables:
        seen_pks = {}
        for c in t.columns:
            if c.missing > 0:
                sev = "Critical" if c.missing_pct > 20 else "Warning"
                issues.append(dict(
                    table=t.name, column=c.name,
                    type="Missing Values", severity=sev,
                    count=c.missing, pct=c.missing_pct,
                    detail=f"{c.missing} nulls ({c.missing_pct}%)",
                    fix=f"Investigate source system for {c.name} — apply data imputation or flag records."
                ))
            if c.is_numeric and c.min_val is not None and c.min_val < 0:
                neg_kw = ["age","bmi","score","count","weight","height","value","result","amount"]
                if any(k in c.name.lower() for k in neg_kw):
                    neg_count = int((t.df[c.name] < 0).sum())
                    if neg_count > 0:
                        issues.append(dict(
                            table=t.name, column=c.name,
                            type="Negative Value", severity="Critical",
                            count=neg_count, pct=round(neg_count/t.row_count*100,2),
                            detail=f"Min value={c.min_val} — biologically implausible negative",
                            fix="Flag for source data verification (SDV) and eCRF correction."
                        ))
            if c.unique_count == 1 and t.row_count > 1:
                issues.append(dict(
                    table=t.name, column=c.name,
                    type="Constant Column", severity="Warning",
                    count=t.row_count, pct=100.0,
                    detail=f"All rows have value: '{c.sample[0] if c.sample else '?'}'",
                    fix="Verify if column is intentionally constant or data load error."
                ))
            # Duplicate PK
            if c.name in t.pk_candidates:
                if c.unique_count < t.row_count:
                    dup = t.row_count - c.unique_count
                    issues.append(dict(
                        table=t.name, column=c.name,
                        type="Duplicate PK", severity="Critical",
                        count=dup, pct=round(dup/t.row_count*100,2),
                        detail=f"{dup} duplicate primary key values",
                        fix="Retract duplicate records and enforce UNIQUE constraint."
                    ))
    return issues

def build_report(tables: list) -> dict:
    rels   = detect_relationships(tables)
    dq     = detect_dq(tables)
    total  = sum(t.row_count for t in tables)
    crit   = [i for i in dq if i["severity"]=="Critical"]
    warns  = [i for i in dq if i["severity"]=="Warning"]
    shared = defaultdict(list)
    for t in tables:
        for c in t.columns:
            shared[_norm(c.name)].append((t.name, c.name))
    shared = {k:v for k,v in shared.items() if len(v)>1}
    return dict(
        tables=tables, relationships=rels, dq_issues=dq,
        shared_cols=dict(shared),
        summary=dict(
            n_tables=len(tables), total_rows=total,
            total_cols=sum(t.col_count for t in tables),
            n_rels=len(rels),
            n_dq=len(dq), n_crit=len(crit), n_warn=len(warns),
            high_conf_rels=len([r for r in rels if r["confidence"]>=70]),
        )
    )

# ─────────────────────────── ER Diagram (SVG) ────────────────────────────────
def render_er_svg(report: dict) -> str:
    tables  = report["tables"]
    rels    = report["relationships"]
    n       = len(tables)
    if n == 0: return ""

    # Layout: arrange tables in a grid
    import math
    cols_grid = min(3, n)
    rows_grid = math.ceil(n / cols_grid)
    W, H = 260, 220
    PAD_X, PAD_Y = 60, 60
    svg_w = cols_grid * W + PAD_X * 2
    svg_h = rows_grid * H + PAD_Y * 2

    positions = {}  # table_name → (cx, cy)
    for idx, t in enumerate(tables):
        col_i = idx % cols_grid
        row_i = idx // cols_grid
        cx = PAD_X + col_i * W + W // 2
        cy = PAD_Y + row_i * H + 60
        positions[t.name] = (cx, cy)

    lines = [
        f'<svg viewBox="0 0 {svg_w} {svg_h}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;background:#f8fafc;border-radius:12px;border:1px solid #e2e8f0">',
        '<defs>',
        '<marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">',
        '<path d="M0,0 L0,6 L9,3 z" fill="#0d9488"/>',
        '</marker>',
        '</defs>',
    ]

    # Draw relationship lines first (behind boxes)
    drawn_rels = set()
    for r in rels[:8]:
        key = (r["from_table"], r["to_table"])
        if key in drawn_rels: continue
        drawn_rels.add(key)
        if r["from_table"] not in positions or r["to_table"] not in positions: continue
        x1,y1 = positions[r["from_table"]]
        x2,y2 = positions[r["to_table"]]
        color = "#0d9488" if r["confidence"]>=70 else "#94a3b8"
        lbl   = f'{r["from_col"]} → {r["to_col"]} ({r["type"]})'
        mx, my = (x1+x2)//2, (y1+y2)//2
        lines.append(
            f'<line x1="{x1}" y1="{y1+30}" x2="{x2}" y2="{y2+30}" '
            f'stroke="{color}" stroke-width="1.5" stroke-dasharray="5,3" '
            f'marker-end="url(#arr)"/>'
        )
        lines.append(
            f'<text x="{mx}" y="{my-6}" text-anchor="middle" '
            f'font-size="9" fill="{color}" font-family="system-ui">{lbl}</text>'
        )

    # Draw table boxes
    for t in tables:
        cx, cy = positions[t.name]
        box_w  = 200
        col_h  = min(len(t.columns), 8) * 16 + 44
        bx     = cx - box_w // 2
        # Shadow
        lines.append(f'<rect x="{bx+3}" y="{cy+3}" width="{box_w}" height="{col_h}" rx="8" fill="#94a3b8" opacity="0.15"/>')
        # Body
        lines.append(f'<rect x="{bx}" y="{cy}" width="{box_w}" height="{col_h}" rx="8" fill="white" stroke="#e2e8f0" stroke-width="1.5"/>')
        # Header
        lines.append(f'<rect x="{bx}" y="{cy}" width="{box_w}" height="28" rx="8" fill="#0d9488"/>')
        lines.append(f'<rect x="{bx}" y="{cy+20}" width="{box_w}" height="8" fill="#0d9488"/>')
        lines.append(
            f'<text x="{cx}" y="{cy+18}" text-anchor="middle" '
            f'font-size="12" font-weight="bold" fill="white" font-family="system-ui">'
            f'🗂 {t.name}</text>'
        )
        lines.append(
            f'<text x="{bx+6}" y="{cy+40}" font-size="8.5" fill="#94a3b8" font-family="system-ui">'
            f'{t.row_count:,} rows · {t.col_count} cols</text>'
        )
        for ci, col in enumerate(t.columns[:7]):
            cy2 = cy + 50 + ci * 16
            icon = "🔑" if col.name in t.pk_candidates else ("📊" if col.is_numeric else "📝")
            dtype_short = "num" if col.is_numeric else ("date" if col.is_date else "str")
            lines.append(
                f'<text x="{bx+8}" y="{cy2}" font-size="9" fill="#334155" font-family="monospace">'
                f'{icon} {col.name[:22]} <tspan fill="#94a3b8">({dtype_short})</tspan></text>'
            )
        if len(t.columns) > 7:
            lines.append(
                f'<text x="{bx+8}" y="{cy+50+7*16}" font-size="8" fill="#94a3b8" font-family="system-ui">'
                f'+ {len(t.columns)-7} more columns…</text>'
            )

    lines.append('</svg>')
    return "\n".join(lines)

# ─────────────────────────── Dashboard (Plotly HTML) ─────────────────────────
COLORS = px.colors.qualitative.Set2 if HAS_PLOTLY else []

def _smart_charts(report: dict, query: str = "") -> str:
    """Generate Plotly multi-panel dashboard as embeddable HTML."""
    if not HAS_PLOTLY:
        return "<p>Plotly not available.</p>"

    tables = report["tables"]
    ql     = query.lower()
    charts = []  # list of (title, fig)

    for t in tables:
        df = t.df

        # Find requested columns from query
        req_cols = [c.name for c in t.columns if c.name.lower() in ql]

        # Numeric columns
        num_cols = [c for c in t.columns if c.is_numeric]
        cat_cols = [c for c in t.columns if not c.is_numeric
                    and not c.is_date and c.unique_count <= 12]

        # --- Distribution charts for numeric cols ---
        for nc in (req_cols[:3] or [c.name for c in num_cols[:3]]):
            col_obj = next((c for c in t.columns if c.name == nc), None)
            if col_obj and col_obj.is_numeric and nc in df.columns:
                fig = px.histogram(
                    df, x=nc, nbins=15,
                    title=f"{t.name} — {nc} Distribution",
                    color_discrete_sequence=["#0d9488"],
                    template="simple_white",
                )
                fig.update_layout(height=280, margin=dict(t=40,b=30,l=40,r=20),
                                  title_font_size=12)
                charts.append(fig)

        # --- Bar charts for categorical cols ---
        for cc in [c for c in cat_cols[:2] if c.name.lower() in ql or not req_cols]:
            if cc.name in df.columns and cc.top_values:
                fig = px.bar(
                    x=list(cc.top_values.keys()),
                    y=list(cc.top_values.values()),
                    title=f"{t.name} — {cc.name} Breakdown",
                    color=list(cc.top_values.keys()),
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    template="simple_white",
                )
                fig.update_layout(height=280, showlegend=False,
                                  margin=dict(t=40,b=30,l=40,r=20),
                                  title_font_size=12)
                charts.append(fig)

        # --- Scatter if two numeric cols mentioned ---
        if len(num_cols) >= 2 and "vs" in ql or "scatter" in ql or "correlation" in ql:
            nc1, nc2 = num_cols[0].name, num_cols[1].name
            if nc1 in df.columns and nc2 in df.columns:
                color_col = cat_cols[0].name if cat_cols else None
                fig = px.scatter(
                    df, x=nc1, y=nc2,
                    color=color_col if color_col and color_col in df.columns else None,
                    title=f"{t.name} — {nc1} vs {nc2}",
                    template="simple_white",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig.update_layout(height=300, margin=dict(t=40,b=30,l=40,r=20),
                                  title_font_size=12)
                charts.append(fig)

        # --- Pie for AE severity or treatment arm ---
        for cc in cat_cols:
            if any(k in cc.name.lower() for k in ["severity","arm","race","sex","country","flag","category"]):
                if cc.name in df.columns and cc.top_values:
                    fig = px.pie(
                        names=list(cc.top_values.keys()),
                        values=list(cc.top_values.values()),
                        title=f"{t.name} — {cc.name}",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                        template="simple_white",
                    )
                    fig.update_layout(height=280, margin=dict(t=40,b=10,l=20,r=20),
                                      title_font_size=12)
                    charts.append(fig)
                    break

        # --- Box plot for response by treatment arm ---
        if "treatment_arm" in [c.name.upper() for c in t.columns]:
            arm_col = next((c.name for c in t.columns if "treatment" in c.name.lower()), None)
            score_col = next((c.name for c in num_cols if "score" in c.name.lower() or "response" in c.name.lower()), None)
            if arm_col and score_col and arm_col in df.columns and score_col in df.columns:
                fig = px.box(
                    df, x=arm_col, y=score_col,
                    color=arm_col,
                    title=f"{t.name} — {score_col} by {arm_col}",
                    color_discrete_sequence=["#0d9488","#2563eb"],
                    template="simple_white",
                )
                fig.update_layout(height=300, showlegend=False,
                                  margin=dict(t=40,b=30,l=40,r=20),
                                  title_font_size=12)
                charts.append(fig)

    # DQ completeness bar
    if report["dq_issues"] or True:
        tnames, comp = [], []
        for t in tables:
            miss_cols = sum(1 for c in t.columns if c.missing > 0)
            pct = round((1 - miss_cols/max(t.col_count,1))*100, 1)
            tnames.append(t.name); comp.append(pct)
        fig = go.Figure(go.Bar(
            x=tnames, y=comp,
            marker_color=["#0d9488" if v>=95 else "#eab308" if v>=80 else "#ef4444" for v in comp],
            text=[f"{v}%" for v in comp], textposition="outside",
        ))
        fig.update_layout(
            title="Data Completeness by Table (%)",
            yaxis=dict(range=[0,110]), template="simple_white",
            height=260, margin=dict(t=40,b=30,l=40,r=20), title_font_size=12,
        )
        charts.append(fig)

    if not charts:
        return "<p style='color:#94a3b8;font-size:.85rem'>No chart data available.</p>"

    # Serialize to HTML
    html_parts = []
    for fig in charts[:8]:
        html_parts.append(fig.to_html(
            full_html=False, include_plotlyjs=False,
            config={"displayModeBar": False, "responsive": True},
        ))

    grid = "".join(
        f'<div style="flex:1 1 340px;min-width:300px;max-width:520px">{h}</div>'
        for h in html_parts
    )
    return (
        '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
        '<div style="display:flex;flex-wrap:wrap;gap:12px;padding:4px 0">'
        + grid + "</div>"
    )

# ─────────────────────────── PDF Report ──────────────────────────────────────
def generate_pdf(report: dict) -> bytes:
    if not HAS_FPDF:
        return b""
    tables = report["tables"]
    rels   = report["relationships"]
    dq     = report["dq_issues"]
    s      = report["summary"]

    def _safe(t):
        txt = str(t)
        subs = {u"—":"--",u"–":"-",u"’":"'",
                u"‘":"'",u"“":'"',u"”":'"',
                u"•":"*",u"²":"2",u"°":"deg"}
        for k,v in subs.items():
            txt = txt.replace(k,v)
        return txt.encode("latin-1","replace").decode("latin-1")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_fill_color(15, 23, 42)
    pdf.rect(0, 0, 210, 28, "F")
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(255,255,255)
    pdf.set_xy(10,8)
    pdf.cell(0, 12, "DataGenome AI - Clinical Data Intelligence Report", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_xy(10,18)
    from datetime import datetime
    pdf.cell(0, 6, _safe("Generated: " + str(datetime.now().strftime("%Y-%m-%d %H:%M")) + " | Tables:" + str(s["n_tables"]) + " | Rows:" + str(s["total_rows"]) + " | DQ:" + str(s["n_dq"]) + " | Rels:" + str(s["n_rels"])), ln=True)
    pdf.set_text_color(0,0,0)
    pdf.ln(8)

    def section(title, color=(13,148,136)):
        pdf.set_fill_color(*color)
        pdf.set_text_color(255,255,255)
        pdf.set_font("Helvetica","B",11)
        pdf.cell(0,8,_safe(f"  {title}"),ln=True,fill=True)
        pdf.set_text_color(0,0,0)
        pdf.ln(2)

    def row_line(label, value, alt=False):
        pdf.set_fill_color(248,250,252) if alt else pdf.set_fill_color(255,255,255)
        pdf.set_font("Helvetica","B",9)
        pdf.cell(70,6,_safe(f"  {label}"),fill=True)
        pdf.set_font("Helvetica","",9)
        pdf.cell(0,6,_safe(str(value)),ln=True,fill=True)

    # Executive summary
    section("1. Executive Summary")
    for i,(lb,vl) in enumerate([
        ("Total Datasets",    s['n_tables']),
        ("Total Records",     f"{s['total_rows']:,}"),
        ("Total Columns",     s['total_cols']),
        ("Relationships Detected", s['n_rels']),
        ("High-Confidence Links", s['high_conf_rels']),
        ("DQ Issues (Total)", s['n_dq']),
        ("  — Critical",      s['n_crit']),
        ("  — Warnings",      s['n_warn']),
    ]):
        row_line(lb, vl, alt=bool(i%2))
    pdf.ln(4)

    # Schema section
    section("2. Schema Intelligence")
    for t in tables:
        pdf.set_font("Helvetica","B",10)
        pdf.set_text_color(13,148,136)
        pdf.cell(0,7,_safe(f"  {t.name}  ({t.row_count} rows x {t.col_count} cols | source: {t.source})"),ln=True)
        pdf.set_text_color(0,0,0)
        pdf.set_font("Helvetica","B",8)
        pdf.set_fill_color(226,232,240)
        for h,w in [("Column",60),("Type",22),("Unique",20),("Missing",20),("Description",68)]:
            pdf.cell(w,6,_safe(h),fill=True)
        pdf.ln()
        for i,c in enumerate(t.columns):
            pdf.set_fill_color(248,250,252) if i%2 else pdf.set_fill_color(255,255,255)
            pdf.set_font("Helvetica","",8)
            dtype = "numeric" if c.is_numeric else ("date" if c.is_date else "text")
            desc  = c.ai_description[:55]+"…" if len(c.ai_description)>55 else c.ai_description
            for val,w in [(c.name[:18],60),(dtype,22),(str(c.unique_count),20),
                          (f"{c.missing}({c.missing_pct}%)",20),(desc,68)]:
                pdf.cell(w,5,_safe(str(val)),fill=True)
            pdf.ln()
        pdf.ln(3)

    # Relationships
    if rels:
        section("3. Entity Relationships")
        pdf.set_font("Helvetica","B",8)
        pdf.set_fill_color(226,232,240)
        for h,w in [("From Table",40),("From Col",38),("To Table",40),("To Col",38),("Type",20),("Confidence",14)]:
            pdf.cell(w,6,_safe(h),fill=True)
        pdf.ln()
        for i,r in enumerate(rels[:15]):
            pdf.set_fill_color(248,250,252) if i%2 else pdf.set_fill_color(255,255,255)
            pdf.set_font("Helvetica","",8)
            for val,w in [(r["from_table"],40),(r["from_col"],38),(r["to_table"],40),(r["to_col"],38),
                          (r["type"],20),(f"{r['confidence']}%",14)]:
                pdf.cell(w,5,_safe(str(val)),fill=True)
            pdf.ln()
        pdf.ln(3)

    # DQ section
    section("4. Data Quality Audit", color=(220,38,38) if s['n_crit']>0 else (202,132,10))
    if not dq:
        pdf.set_font("Helvetica","",9)
        pdf.cell(0,6,"  No data quality issues detected.",ln=True)
    else:
        pdf.set_font("Helvetica","B",8)
        pdf.set_fill_color(226,232,240)
        for h,w in [("Severity",22),("Table",35),("Column",38),("Issue",30),("Count",15),("Detail",50)]:
            pdf.cell(w,6,_safe(h),fill=True)
        pdf.ln()
        for i,iss in enumerate(dq):
            pdf.set_fill_color(255,235,235) if iss["severity"]=="Critical" else pdf.set_fill_color(255,253,235) if i%2 else pdf.set_fill_color(255,255,255)
            pdf.set_font("Helvetica","",8)
            for val,w in [(iss["severity"],22),(iss["table"],35),(iss["column"],38),
                          (iss["type"],30),(str(iss["count"]),15),(iss["detail"][:28],50)]:
                pdf.cell(w,5,_safe(str(val)),fill=True)
            pdf.ln()
        pdf.ln(3)

    # Data Dictionary
    section("5. AI Data Dictionary")
    pdf.set_font("Helvetica","B",8)
    pdf.set_fill_color(226,232,240)
    for h,w in [("Table",35),("Column",50),("Type",20),("AI Description",85)]:
        pdf.cell(w,6,_safe(h),fill=True)
    pdf.ln()
    for i,t in enumerate(tables):
        for j,c in enumerate(t.columns):
            alt = (i+j)%2==0
            pdf.set_fill_color(248,250,252) if alt else pdf.set_fill_color(255,255,255)
            pdf.set_font("Helvetica","",7.5)
            dtype = "numeric" if c.is_numeric else ("date" if c.is_date else "text")
            desc = c.ai_description[:60]+"…" if len(c.ai_description)>60 else c.ai_description
            for val,w in [(t.name[:14],35),(c.name[:20],50),(dtype,20),(desc,85)]:
                pdf.cell(w,5,_safe(str(val)),fill=True)
            pdf.ln()
    pdf.ln(3)

    # Glossary
    section("6. Life Sciences Business Glossary")
    all_cols = {c.name.upper() for t in tables for c in t.columns}
    rel_terms = {k:v for k,v in LS_GLOSSARY.items() if k.upper() in all_cols or len(k)<8}
    pdf.set_font("Helvetica","B",8)
    pdf.set_fill_color(226,232,240)
    pdf.cell(45,6,"Term",fill=True); pdf.cell(145,6,"Definition",fill=True,ln=True)
    for i,(term,defn) in enumerate(list(rel_terms.items())[:30]):
        pdf.set_fill_color(248,250,252) if i%2 else pdf.set_fill_color(255,255,255)
        pdf.set_font("Helvetica","",8)
        pdf.cell(45,5,_safe(str(term)[:20]),fill=True)
        pdf.multi_cell(145,5,_safe(str(defn)[:90]),fill=True)

    return pdf.output()

# ─────────────────────────── Data Dictionary CSV ─────────────────────────────
def dict_to_csv(report: dict) -> str:
    rows = []
    for t in report["tables"]:
        for c in t.columns:
            dtype = "numeric" if c.is_numeric else ("date" if c.is_date else "text")
            pk    = "YES" if c.name in t.pk_candidates else "NO"
            rows.append({
                "Table": t.name, "Column": c.name, "Data_Type": dtype,
                "Is_PK": pk, "Unique_Values": c.unique_count,
                "Missing_Count": c.missing, "Missing_Pct": c.missing_pct,
                "Min": c.min_val, "Max": c.max_val,
                "Mean": c.mean_val, "Sample_Values": "|".join(c.sample[:3]),
                "AI_Description": c.ai_description,
            })
    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return buf.getvalue()

# ─────────────────────────── BM25 RAG ────────────────────────────────────────
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

def _tok(text): return re.findall(r'[a-z0-9_]+', text.lower())

def build_rag(report: dict):
    if not HAS_BM25:
        return None, []
    chunks = []
    s = report["summary"]
    chunks.append({"id":"overview","cat":"overview","title":"Dataset Overview",
        "text":f"overview {s['n_tables']} tables {s['total_rows']} rows {s['total_cols']} columns {s['n_rels']} relationships {s['n_dq']} dq issues",
        "data":s})
    for t in report["tables"]:
        col_names = " ".join(c.name for c in t.columns)
        chunks.append({"id":f"tbl_{t.name}","cat":"schema","title":f"Table {t.name}",
            "text":f"table {t.name} {t.source} {t.row_count} rows {t.col_count} columns {col_names} {t.description}",
            "data":t})
        for c in t.columns:
            stat = f"min {c.min_val} max {c.max_val} mean {c.mean_val}" if c.is_numeric else ""
            tops = " ".join(str(k) for k in list(c.top_values.keys())[:6])
            chunks.append({"id":f"col_{t.name}_{c.name}","cat":"column","title":f"{t.name}.{c.name}",
                "text":f"column {c.name} table {t.name} {c.dtype} numeric {c.is_numeric} unique {c.unique_count} missing {c.missing} {stat} {tops} {c.ai_description}",
                "data":(t.name, c)})
    for i,iss in enumerate(report["dq_issues"]):
        chunks.append({"id":f"dq_{i}","cat":"dq","title":f"DQ {iss['table']}.{iss['column']}",
            "text":f"data quality issue {iss['table']} {iss['column']} {iss['type']} {iss['severity']} {iss['count']} {iss['detail']} {iss['fix']}",
            "data":iss})
    for i,r in enumerate(report["relationships"]):
        chunks.append({"id":f"rel_{i}","cat":"relationship","title":f"Rel {r['from_table']} {r['to_table']}",
            "text":f"relationship join link {r['from_table']} {r['from_col']} to {r['to_table']} {r['to_col']} {r['type']} confidence {r['confidence']} er diagram foreign key",
            "data":r})
    # Glossary chunks
    for term, defn in LS_GLOSSARY.items():
        chunks.append({"id":f"g_{term}","cat":"glossary","title":f"Glossary: {term}",
            "text":f"glossary term {term} definition {defn} life science clinical",
            "data":{"term":term,"definition":defn}})
    idx = BM25Okapi([_tok(c["text"]+" "+c["title"]) for c in chunks])
    return idx, chunks

def retrieve(idx, chunks, query: str, top_k=6):
    if idx is None: return []
    scores = idx.get_scores(_tok(query))
    ranked = sorted([(i,s) for i,s in enumerate(scores) if s>0],
                    key=lambda x:-x[1])[:top_k]
    return [{**chunks[i],"score":round(float(s),3)} for i,s in ranked]
