"""
chart_engine.py — Plotly chart generation for DataGenome AI
Produces rich, interactive Streamlit-native charts from DatasetInfo + user queries.
Zero cost, no API, pure pandas + plotly.
"""
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TEAL   = "#0d9488"
COLORS = ["#0d9488","#2563eb","#7c3aed","#ca8a04","#dc2626",
          "#0369a1","#15803d","#c026d3","#ea580c","#0284c7"]

# ── Intent parsing ────────────────────────────────────────────────────────────
def _tok(t): return re.findall(r'[a-z0-9_]+', t.lower())

CHART_WORDS = ["dashboard","chart","plot","graph","visual","show","distribution",
               "compare","versus","vs","breakdown","histogram","bar","pie",
               "scatter","trend","correlation","boxplot","box plot","heatmap",
               "frequency","analyse","analyze","profile"]

def is_chart_query(query: str) -> bool:
    ql = query.lower()
    return any(w in ql for w in CHART_WORDS)

def _find_cols_in_query(query: str, df: pd.DataFrame) -> list:
    """Return list of column names mentioned in query (case-insensitive)."""
    ql = query.lower()
    found = []
    for col in df.columns:
        if col.lower() in ql or col.lower().replace("_"," ") in ql:
            found.append(col)
    # Also match short aliases  e.g. "age" → "AGE"
    for col in df.columns:
        tok = col.lower().lstrip("_")
        if tok in _tok(ql) and col not in found:
            found.append(col)
    return found

def _find_table_in_query(query: str, datasets: list):
    ql = query.lower()
    for ds in datasets:
        if ds.name.lower() in ql:
            return ds
    return None

def _is_numeric(series: pd.Series) -> bool:
    return pd.to_numeric(series, errors="coerce").notna().sum() > len(series) * 0.5

def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

# ── Auto key-column detection ─────────────────────────────────────────────────
def detect_key_columns(df: pd.DataFrame) -> dict:
    """
    Automatically classify columns by role.
    Returns dict with keys: numeric, categorical, datetime, id_cols, target_candidates
    """
    numeric, categorical, datetime_cols, id_cols = [], [], [], []

    for col in df.columns:
        s = df[col]
        cl = col.lower()

        # ID pattern
        if any(k in cl for k in ["id","key","code","seq","num","no"]) and s.nunique() > len(df)*0.8:
            id_cols.append(col); continue

        # Datetime
        try:
            if s.dtype == "object":
                parsed = pd.to_datetime(s, errors="coerce")
                if parsed.notna().sum() > len(s) * 0.5:
                    datetime_cols.append(col); continue
        except Exception:
            pass

        num = _to_num(s)
        if num.notna().sum() > len(s) * 0.5:
            numeric.append(col)
        else:
            if s.nunique() <= max(20, len(df) * 0.3):
                categorical.append(col)

    # Target candidates = numeric cols likely to be outcomes/scores
    target_kw = ["score","result","value","rate","count","amount",
                 "response","outcome","measure","level","concentration"]
    targets = [c for c in numeric
               if any(k in c.lower() for k in target_kw)]

    return {
        "numeric":      numeric,
        "categorical":  categorical,
        "datetime":     datetime_cols,
        "id_cols":      id_cols,
        "targets":      targets,
    }


# ── Individual chart builders ─────────────────────────────────────────────────
def _hist(df, col, color=None, title=None):
    num = _to_num(df[col]).dropna()
    if color and color in df.columns:
        fig = px.histogram(df.dropna(subset=[col]), x=col, color=color,
                           barmode="overlay", opacity=0.75,
                           color_discrete_sequence=COLORS,
                           title=title or f"Distribution of {col} by {color}")
    else:
        fig = px.histogram(df, x=col, nbins=min(30, max(10, len(num)//3)),
                           color_discrete_sequence=[TEAL],
                           title=title or f"Distribution of {col}")
        fig.add_vline(x=float(num.mean()), line_dash="dash",
                      line_color="#dc2626", annotation_text=f"Mean={num.mean():.1f}")
        fig.add_vline(x=float(num.median()), line_dash="dot",
                      line_color="#7c3aed", annotation_text=f"Median={num.median():.1f}")
    fig.update_layout(**_layout())
    return fig

def _bar_cat(df, col, color=None, title=None, top_n=15):
    vc = df[col].value_counts().head(top_n)
    if color and color in df.columns:
        fig = px.histogram(df, x=col, color=color, barmode="group",
                           color_discrete_sequence=COLORS,
                           category_orders={col: vc.index.tolist()},
                           title=title or f"{col} by {color}")
    else:
        fig = px.bar(x=vc.index, y=vc.values,
                     color=vc.values, color_continuous_scale="teal",
                     title=title or f"Frequency of {col}",
                     labels={"x": col, "y": "Count"})
        fig.update_coloraxes(showscale=False)
    fig.update_layout(**_layout())
    return fig

def _pie(df, col, title=None, top_n=8):
    vc = df[col].value_counts().head(top_n)
    fig = px.pie(values=vc.values, names=vc.index,
                 color_discrete_sequence=COLORS,
                 title=title or f"{col} — Proportion",
                 hole=0.35)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(**_layout())
    return fig

def _box(df, cat_col, num_col, title=None):
    fig = px.box(df.dropna(subset=[cat_col, num_col]),
                 x=cat_col, y=num_col, color=cat_col,
                 color_discrete_sequence=COLORS,
                 title=title or f"{num_col} by {cat_col}",
                 points="outliers")
    fig.update_layout(**_layout())
    return fig

def _scatter(df, x_col, y_col, color=None, title=None):
    kw = dict(color=color, color_discrete_sequence=COLORS) if color else {}
    fig = px.scatter(df.dropna(subset=[x_col, y_col]),
                     x=x_col, y=y_col,
                     trendline="ols", trendline_color_override="#dc2626",
                     opacity=0.7,
                     title=title or f"{x_col} vs {y_col}",
                     **kw)
    fig.update_layout(**_layout())
    return fig

def _corr_heatmap(df, cols, title="Correlation Matrix"):
    num_df = df[cols].apply(pd.to_numeric, errors="coerce").dropna(how="all", axis=1)
    if num_df.shape[1] < 2:
        return None
    corr = num_df.corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu", zmid=0, text=corr.values,
        texttemplate="%{text}", textfont={"size":10},
        hovertemplate="%{x} vs %{y}: %{z}<extra></extra>",
    ))
    fig.update_layout(title=title, **_layout())
    return fig

def _line(df, x_col, y_col, color=None, title=None):
    kw = dict(color=color, color_discrete_sequence=COLORS) if color else {}
    agg = df.groupby(x_col)[y_col].mean().reset_index() if not color else df
    fig = px.line(agg, x=x_col, y=y_col,
                  markers=True,
                  color_discrete_sequence=[TEAL],
                  title=title or f"{y_col} over {x_col}",
                  **kw)
    fig.update_layout(**_layout())
    return fig

def _violin(df, cat_col, num_col, title=None):
    fig = px.violin(df.dropna(subset=[cat_col, num_col]),
                    x=cat_col, y=num_col, color=cat_col,
                    box=True, points="all",
                    color_discrete_sequence=COLORS,
                    title=title or f"{num_col} distribution by {cat_col}")
    fig.update_layout(**_layout())
    return fig

def _missing_bar(df, title="Missing Values"):
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    if miss.empty:
        return None
    pct = (miss / len(df) * 100).round(1)
    fig = px.bar(x=miss.index, y=pct.values,
                 color=pct.values, color_continuous_scale="Reds",
                 title=title,
                 labels={"x": "Column", "y": "% Missing"})
    fig.update_coloraxes(showscale=False)
    fig.update_layout(**_layout())
    return fig

def _layout():
    return dict(
        height=360,
        margin=dict(l=40, r=20, t=50, b=40),
        paper_bgcolor="white",
        plot_bgcolor="#f8fafc",
        font=dict(family="system-ui, sans-serif", size=12, color="#1e293b"),
        title_font=dict(size=14, color="#0f172a", family="system-ui, sans-serif"),
        legend=dict(bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#e2e8f0", borderwidth=1),
        xaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0"),
        yaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0"),
    )

# ── Dashboard builders ────────────────────────────────────────────────────────
def build_two_col_dashboard(df, col1, col2, ds_name="") -> list:
    """
    Smart dashboard comparing two columns.
    Returns list of (title, plotly_fig) tuples.
    """
    charts = []
    c1_num = _is_numeric(df[col1])
    c2_num = _is_numeric(df[col2])

    if c1_num and c2_num:
        # Numeric vs Numeric
        charts.append((f"{col1} vs {col2} — Scatter", _scatter(df, col1, col2)))
        charts.append((f"Distribution of {col1}", _hist(df, col1)))
        charts.append((f"Distribution of {col2}", _hist(df, col2)))
        charts.append((f"Correlation Heatmap", _corr_heatmap(df, [col1, col2])))

    elif c1_num and not c2_num:
        # Numeric vs Categorical
        charts.append((f"{col1} by {col2}", _box(df, col2, col1)))
        charts.append((f"{col1} violin by {col2}", _violin(df, col2, col1)))
        charts.append((f"Distribution of {col1}", _hist(df, col1, color=col2)))
        charts.append((f"Frequency of {col2}", _bar_cat(df, col2)))

    elif not c1_num and c2_num:
        # Categorical vs Numeric
        charts.append((f"{col2} by {col1}", _box(df, col1, col2)))
        charts.append((f"{col2} violin by {col1}", _violin(df, col1, col2)))
        charts.append((f"Distribution of {col2}", _hist(df, col2, color=col1)))
        charts.append((f"Frequency of {col1}", _bar_cat(df, col1)))

    else:
        # Categorical vs Categorical
        charts.append((f"{col1} proportion", _pie(df, col1)))
        charts.append((f"{col2} proportion", _pie(df, col2)))
        charts.append((f"{col1} by {col2}", _bar_cat(df, col1, color=col2)))
        charts.append((f"{col2} by {col1}", _bar_cat(df, col2, color=col1)))

    return [(t, f) for t, f in charts if f is not None]


def build_single_col_dashboard(df, col, ds_name="") -> list:
    """Smart single-column dashboard."""
    charts = []
    if _is_numeric(df[col]):
        charts.append((f"Distribution — {col}", _hist(df, col)))
        # Box by any available categorical
        cats = [c for c in df.columns
                if not _is_numeric(df[c]) and df[c].nunique() <= 10 and c != col]
        for cat in cats[:2]:
            charts.append((f"{col} by {cat}", _box(df, cat, col)))
    else:
        charts.append((f"Frequency — {col}", _bar_cat(df, col)))
        charts.append((f"Proportion — {col}", _pie(df, col)))
        # Breakdown by other categoricals
        cats = [c for c in df.columns
                if not _is_numeric(df[c]) and df[c].nunique() <= 10
                and c != col and df[c].nunique() >= 2]
        for cat in cats[:1]:
            charts.append((f"{col} × {cat}", _bar_cat(df, col, color=cat)))
    return [(t, f) for t, f in charts if f is not None]


def build_auto_key_dashboard(df, ds_name="") -> list:
    """
    Automatically analyse all key columns — no user selection needed.
    Returns list of (title, fig) tuples for the most insightful charts.
    """
    key = detect_key_columns(df)
    charts = []
    num_cols = key["numeric"][:6]
    cat_cols = key["categorical"][:4]

    # 1. Correlation heatmap for all numeric cols
    if len(num_cols) >= 2:
        fig = _corr_heatmap(df, num_cols, "Numeric Columns — Correlation Matrix")
        if fig: charts.append(("Correlation Matrix", fig))

    # 2. Distribution for each numeric
    for col in num_cols[:4]:
        # Colour by first categorical if available
        color = cat_cols[0] if cat_cols else None
        charts.append((f"Distribution — {col}", _hist(df, col, color=color)))

    # 3. Frequency/pie for each categorical
    for col in cat_cols[:4]:
        charts.append((f"Breakdown — {col}", _bar_cat(df, col)))

    # 4. Box: first numeric vs each categorical
    if num_cols and cat_cols:
        for cat in cat_cols[:3]:
            charts.append((f"{num_cols[0]} by {cat}", _box(df, cat, num_cols[0])))

    # 5. Scatter: first two numerics
    if len(num_cols) >= 2:
        color = cat_cols[0] if cat_cols else None
        charts.append((f"{num_cols[0]} vs {num_cols[1]}",
                       _scatter(df, num_cols[0], num_cols[1], color=color)))

    # 6. Missing values
    fig = _missing_bar(df, f"Missing Values — {ds_name}")
    if fig: charts.append(("Missing Values", fig))

    return [(t, f) for t, f in charts if f is not None]


def build_cross_dataset_dashboard(ds_list: list) -> list:
    """
    Cross-dataset comparative dashboard.
    Finds shared columns and plots them side by side.
    """
    from analyzer import _normalize
    charts = []

    # Find shared numeric columns
    col_sets = []
    for ds in ds_list:
        cols = {_normalize(c.name): (ds, c.name)
                for c in ds.columns if c.is_numeric}
        col_sets.append(cols)

    # Common columns across all datasets
    if len(col_sets) >= 2:
        common = set(col_sets[0].keys())
        for cs in col_sets[1:]:
            common &= set(cs.keys())

        for norm_col in list(common)[:4]:
            frames = []
            for ds in ds_list:
                for c in ds.columns:
                    if _normalize(c.name) == norm_col:
                        tmp = ds.df[[c.name]].copy()
                        tmp = pd.to_numeric(tmp[c.name], errors="coerce").dropna()
                        frame = pd.DataFrame({c.name: tmp, "Dataset": ds.name})
                        frames.append(frame)
            if frames:
                combined = pd.concat(frames, ignore_index=True)
                col_name = frames[0].columns[0]
                fig = px.box(combined, x="Dataset", y=col_name,
                             color="Dataset", color_discrete_sequence=COLORS,
                             title=f"{col_name} — Cross-dataset Comparison",
                             points="all")
                fig.update_layout(**_layout())
                charts.append((f"Compare: {col_name}", fig))

    return charts


# ── Main dispatcher ───────────────────────────────────────────────────────────
def resolve_chart_request(query: str, datasets: list) -> dict:
    """
    Parse query, find relevant dataset & columns, return chart bundle.
    Returns: {
        "title": str,
        "charts": [(title, fig), ...],
        "summary": str,
        "ds_name": str,
    }
    """
    ql = query.lower()
    ds = _find_table_in_query(query, datasets)

    # Default to first dataset if none named
    if ds is None and datasets:
        ds = datasets[0]
    if ds is None:
        return {"title": "No data", "charts": [], "summary": "No datasets loaded.", "ds_name": ""}

    df = ds.df
    cols_found = _find_cols_in_query(query, df)

    # Cross-dataset compare
    if len(datasets) > 1 and any(w in ql for w in ["cross","compare","between dataset","all dataset"]):
        charts = build_cross_dataset_dashboard(datasets)
        return {
            "title": "Cross-Dataset Comparison",
            "charts": charts,
            "summary": f"Comparing shared columns across {len(datasets)} datasets.",
            "ds_name": "All datasets",
        }

    # Auto key-column dashboard
    if not cols_found or any(w in ql for w in ["auto","key","all col","full","overview","profile","automatic"]):
        charts = build_auto_key_dashboard(df, ds.name)
        return {
            "title": f"Auto Dashboard — {ds.name}",
            "charts": charts,
            "summary": f"Automatic analysis of key columns in {ds.name} ({ds.row_count:,} rows).",
            "ds_name": ds.name,
        }

    # Two specific columns
    if len(cols_found) >= 2:
        col1, col2 = cols_found[0], cols_found[1]
        charts = build_two_col_dashboard(df, col1, col2, ds.name)
        return {
            "title": f"Dashboard — {col1} & {col2} in {ds.name}",
            "charts": charts,
            "summary": f"Comparative analysis of {col1} and {col2} in {ds.name}.",
            "ds_name": ds.name,
        }

    # Single column
    if len(cols_found) == 1:
        col = cols_found[0]
        charts = build_single_col_dashboard(df, col, ds.name)
        return {
            "title": f"Dashboard — {col} in {ds.name}",
            "charts": charts,
            "summary": f"Analysis of {col} in {ds.name}.",
            "ds_name": ds.name,
        }

    # Fallback: auto dashboard
    charts = build_auto_key_dashboard(df, ds.name)
    return {
        "title": f"Auto Dashboard — {ds.name}",
        "charts": charts,
        "summary": f"Key column analysis for {ds.name}.",
        "ds_name": ds.name,
    }
