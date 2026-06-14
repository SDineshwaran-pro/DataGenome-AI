"""
dynamic_rag.py — Builds BM25 RAG index from user-uploaded datasets
Replaces / extends the static rag_engine with live user data.
Handles natural language queries against uploaded datasets.
"""
import re
from rank_bm25 import BM25Okapi
from data_loader import DatasetInfo, ColumnInfo

def _tok(text: str) -> list:
    return re.findall(r'[a-z0-9_\.]+', text.lower())

# ── Chunk builder from DatasetInfo ────────────────────────────────────────────
def build_chunks_from_report(report: dict) -> list:
    """Convert analysis report into BM25-searchable chunks."""
    chunks = []
    datasets = report["datasets"]

    # 1. Overview chunk
    s = report["summary"]
    chunks.append({
        "id": "overview", "category": "overview",
        "title": "Dataset Collection Overview",
        "text": (
            f"uploaded datasets {s['total_datasets']} "
            f"total rows {s['total_rows']} total columns {s['total_cols']} "
            f"relationships found {s['total_relationships']} "
            f"high confidence links {s['high_conf_rels']} "
            f"data quality issues {s['total_dq_issues']} "
            f"critical {s['critical_dq']} warnings {s['warning_dq']} "
            f"shared columns across tables {s['shared_col_groups']}"
        ),
        "data": s,
    })

    # 2. Per-dataset table chunks
    for ds in datasets:
        col_names = " ".join(c.name for c in ds.columns)
        num_cols  = " ".join(c.name for c in ds.columns if c.is_numeric)
        cat_cols  = " ".join(c.name for c in ds.columns if not c.is_numeric)
        chunks.append({
            "id": f"table_{ds.name}", "category": "schema",
            "title": f"Table {ds.name} ({ds.source})",
            "text": (
                f"table {ds.name} source {ds.source} "
                f"rows {ds.row_count} columns {ds.col_count} "
                f"description {ds.description} "
                f"all columns {col_names} "
                f"numeric columns {num_cols} "
                f"categorical columns {cat_cols}"
            ),
            "data": {"name": ds.name, "row_count": ds.row_count,
                     "col_count": ds.col_count, "source": ds.source,
                     "description": ds.description},
        })

    # 3. Per-column chunks
    for ds in datasets:
        for c in ds.columns:
            stat_text = ""
            if c.is_numeric:
                stat_text = (
                    f"min {c.min_val} max {c.max_val} "
                    f"mean {c.mean_val} median {c.median_val} std {c.std_val} "
                    f"maximum highest largest {c.max_val} "
                    f"minimum lowest smallest {c.min_val} "
                    f"average {c.mean_val} "
                )
            top_vals = " ".join(str(k) for k in list(c.top_values.keys())[:6])
            chunks.append({
                "id": f"col_{ds.name}_{c.name}", "category": "column",
                "title": f"Column {ds.name}.{c.name}",
                "text": (
                    f"column {c.name} table {ds.name} "
                    f"type {c.dtype} numeric {c.is_numeric} "
                    f"unique values {c.unique_count} "
                    f"missing {c.missing} missing percent {c.missing_pct} "
                    f"sample {' '.join(c.sample)} "
                    f"top values {top_vals} "
                    f"{stat_text}"
                ),
                "data": c,
            })

    # 4. DQ issue chunks
    for i, issue in enumerate(report["dq_issues"]):
        chunks.append({
            "id": f"dq_{i}", "category": "dq",
            "title": f"DQ Issue — {issue['table']}.{issue['column']} ({issue['severity']})",
            "text": (
                f"data quality issue table {issue['table']} "
                f"column {issue['column']} type {issue['type']} "
                f"severity {issue['severity']} count {issue['count']} "
                f"percent {issue['pct']} detail {issue['detail']}"
            ),
            "data": issue,
        })

    # 5. Relationship chunks
    for r in report["relationships"]:
        chunks.append({
            "id": f"rel_{r['from_table']}_{r['from_col']}_{r['to_table']}_{r['to_col']}",
            "category": "relationship",
            "title": f"Relationship: {r['from_table']}.{r['from_col']} → {r['to_table']}.{r['to_col']}",
            "text": (
                f"relationship join link {r['from_table']} {r['from_col']} "
                f"to {r['to_table']} {r['to_col']} "
                f"type {r['type']} confidence {r['confidence']} "
                f"overlap {r['overlap_pct']} percent "
                f"foreign key entity relationship er diagram"
            ),
            "data": r,
        })

    # 6. Shared columns chunk
    if report["shared_cols"]:
        shared_text = " ".join(
            f"{k} appears in {' '.join(t for t,_ in v)}"
            for k, v in list(report["shared_cols"].items())[:20]
        )
        chunks.append({
            "id": "shared_columns", "category": "schema",
            "title": "Shared Columns Across Datasets",
            "text": f"shared columns common fields across tables {shared_text}",
            "data": report["shared_cols"],
        })

    return chunks


# ── Dynamic RAG index ─────────────────────────────────────────────────────────
class DynamicRAG:
    def __init__(self, report: dict):
        self.report  = report
        self.chunks  = build_chunks_from_report(report)
        self._index  = BM25Okapi([
            _tok(c["text"] + " " + c["title"]) for c in self.chunks
        ])

    def retrieve(self, query: str, top_k: int = 6) -> list:
        tokens = _tok(query)
        if not tokens:
            return []
        scores = self._index.get_scores(tokens)
        ranked = sorted(
            [(i, s) for i, s in enumerate(scores) if s > 0],
            key=lambda x: x[1], reverse=True,
        )[:top_k]
        results = []
        for idx, score in ranked:
            c = dict(self.chunks[idx])
            c["score"] = round(float(score), 3)
            results.append(c)
        return results

    def stats(self) -> dict:
        by_cat = {}
        for c in self.chunks:
            by_cat[c["category"]] = by_cat.get(c["category"], 0) + 1
        return {"total": len(self.chunks), "by_category": by_cat}
