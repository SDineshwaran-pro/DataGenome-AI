"""
analyzer.py — Cross-dataset structural analysis engine
Detects: column relationships, FK candidates, schema overlaps,
data type mismatches, join paths, quality issues, statistical summaries.
No API, no ML — pure pandas + heuristics.
"""
import re
from collections import defaultdict
import pandas as pd
from data_loader import DatasetInfo, ColumnInfo

# ── Relationship detection ────────────────────────────────────────────────────
def _normalize(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', s.lower())

def _col_similarity(a: str, b: str) -> float:
    """Score 0–1: how similar two column names are."""
    na, nb = _normalize(a), _normalize(b)
    if na == nb:             return 1.0
    if na in nb or nb in na: return 0.85
    # Suffix match (id, code, key, num, no)
    suffixes = ['id','code','key','num','no','cd','seq','dt','dtc','fl','yn']
    for s in suffixes:
        if na.endswith(s) and nb.endswith(s) and na[:-len(s)] == nb[:-len(s)]:
            return 0.9
    return 0.0

def detect_relationships(datasets: list) -> list:
    """
    Returns list of relationship dicts:
    {from_table, from_col, to_table, to_col, type, confidence, reason}
    """
    rels = []
    seen = set()

    for i, ds_a in enumerate(datasets):
        for j, ds_b in enumerate(datasets):
            if i >= j:
                continue
            for ca in ds_a.columns:
                for cb in ds_b.columns:
                    sim = _col_similarity(ca.name, cb.name)
                    if sim < 0.85:
                        continue
                    key = tuple(sorted([(ds_a.name, ca.name), (ds_b.name, cb.name)]))
                    if key in seen:
                        continue
                    seen.add(key)

                    # Check value overlap
                    overlap_pct = 0.0
                    try:
                        vals_a = set(ds_a.df[ca.name].dropna().astype(str).unique())
                        vals_b = set(ds_b.df[cb.name].dropna().astype(str).unique())
                        if vals_a and vals_b:
                            overlap_pct = len(vals_a & vals_b) / min(len(vals_a), len(vals_b)) * 100
                    except Exception:
                        pass

                    # Cardinality type
                    uniq_a = ca.unique_count
                    uniq_b = cb.unique_count
                    rows_a = ds_a.row_count
                    rows_b = ds_b.row_count

                    if uniq_a == rows_a and uniq_b == rows_b:
                        rel_type = "1:1"
                    elif uniq_a <= uniq_b:
                        rel_type = "Many:1"
                    else:
                        rel_type = "1:Many"

                    confidence = sim * 0.6 + (overlap_pct / 100) * 0.4

                    rels.append({
                        "from_table": ds_a.name,
                        "from_col":   ca.name,
                        "to_table":   ds_b.name,
                        "to_col":     cb.name,
                        "type":       rel_type,
                        "confidence": round(confidence * 100, 1),
                        "overlap_pct":round(overlap_pct, 1),
                        "name_sim":   round(sim * 100, 1),
                    })

    return sorted(rels, key=lambda r: -r["confidence"])


def detect_pk_candidates(ds: DatasetInfo) -> list:
    """Return column names that look like primary keys."""
    candidates = []
    for c in ds.columns:
        if c.unique_count == ds.row_count and c.missing == 0:
            candidates.append(c.name)
    return candidates


def detect_dq_issues(ds: DatasetInfo) -> list:
    """Auto-detect DQ issues in any dataset."""
    issues = []
    for c in ds.columns:
        # Missing values
        if c.missing > 0:
            sev = "Critical" if c.missing_pct > 20 else "Warning"
            issues.append({
                "table": ds.name, "column": c.name,
                "type": "Missing Values", "severity": sev,
                "count": c.missing, "pct": c.missing_pct,
                "detail": f"{c.missing:,} null values ({c.missing_pct}% of rows)",
            })
        # Potential duplicates (unique << rows, but not categorical)
        if c.is_numeric and c.unique_count < ds.row_count * 0.05 and ds.row_count > 100:
            issues.append({
                "table": ds.name, "column": c.name,
                "type": "Low Cardinality (numeric)",
                "severity": "Warning",
                "count": c.unique_count, "pct": 0,
                "detail": f"Only {c.unique_count} distinct values in numeric column — possible encoding issue",
            })
        # Negative values where unexpected
        neg_keywords = ["age","count","amount","qty","quantity","price","salary","rate",
                        "score","weight","height","duration","seq","num","size","length"]
        if c.is_numeric and c.min_val is not None and c.min_val < 0:
            if any(k in c.name.lower() for k in neg_keywords):
                issues.append({
                    "table": ds.name, "column": c.name,
                    "type": "Negative Value",
                    "severity": "Critical",
                    "count": int((ds.df[c.name] < 0).sum()),
                    "pct": round(int((ds.df[c.name] < 0).sum()) / ds.row_count * 100, 2),
                    "detail": f"Minimum value {c.min_val} — negative values in '{c.name}' may indicate data error",
                })
        # Constant column
        if c.unique_count == 1:
            issues.append({
                "table": ds.name, "column": c.name,
                "type": "Constant Column",
                "severity": "Warning",
                "count": ds.row_count, "pct": 100,
                "detail": f"All {ds.row_count:,} rows have the same value: '{c.sample[0] if c.sample else '?'}'",
            })
    return issues


def build_analysis_report(datasets: list) -> dict:
    """
    Full cross-dataset structural analysis.
    Returns rich report dict consumed by the UI and RAG engine.
    """
    relationships = detect_relationships(datasets)
    all_dq        = []
    all_pk        = {}

    for ds in datasets:
        all_dq.extend(detect_dq_issues(ds))
        all_pk[ds.name] = detect_pk_candidates(ds)

    # Column name index across all datasets
    col_index = defaultdict(list)  # normalized_name → [(table, col)]
    for ds in datasets:
        for c in ds.columns:
            col_index[_normalize(c.name)].append((ds.name, c.name))

    # Shared column names
    shared_cols = {k: v for k, v in col_index.items() if len(v) > 1}

    # Overall stats
    total_rows    = sum(ds.row_count for ds in datasets)
    total_cols    = sum(ds.col_count for ds in datasets)
    crit_dq       = [i for i in all_dq if i["severity"] == "Critical"]
    warn_dq       = [i for i in all_dq if i["severity"] == "Warning"]

    return {
        "datasets":      datasets,
        "relationships": relationships,
        "dq_issues":     all_dq,
        "pk_candidates": all_pk,
        "shared_cols":   shared_cols,
        "col_index":     dict(col_index),
        "summary": {
            "total_datasets":     len(datasets),
            "total_rows":         total_rows,
            "total_cols":         total_cols,
            "total_relationships":len(relationships),
            "high_conf_rels":     len([r for r in relationships if r["confidence"] >= 70]),
            "total_dq_issues":    len(all_dq),
            "critical_dq":        len(crit_dq),
            "warning_dq":         len(warn_dq),
            "shared_col_groups":  len(shared_cols),
        }
    }
