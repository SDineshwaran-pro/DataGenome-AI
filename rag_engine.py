"""
rag_engine.py  —  Zero-cost RAG Engine
BM25 retrieval + template-driven answer synthesis
No API key · No internet · Fully local · Runs on Streamlit Cloud free tier
"""

import re
from rank_bm25 import BM25Okapi
from data import CLINICAL_TABLES, GLOSSARY, MOCK_DQ_ISSUES

# ─────────────────────────── Tokenizer ───────────────────────────────────────
def _tok(text: str) -> list:
    return re.findall(r'[a-z0-9_]+', text.lower())

# ─────────────────────────── Chunk Builder ───────────────────────────────────
def _build_chunks() -> list:
    chunks = []

    # 1. Study overview
    chunks.append({
        "id": "study_overview", "category": "study",
        "title": "Study Overview — STUDY-2026-FIBER",
        "text": (
            "Study ID STUDY-2026-FIBER Phase III RCT Drug X vs Placebo "
            "Chronic Inflammatory Disease 250 subjects randomized 1 to 1 "
            "primary endpoint reduction disease severity score Week 24 "
            "data cutoff May 2026 CDISC SDTM ADaM FDA EMA regulatory submission "
            "total records 10240 sponsor DataGenome Pharma"
        ),
        "data": {
            "study_id": "STUDY-2026-FIBER", "phase": "III", "subjects": 250,
            "indication": "Chronic Inflammatory Disease",
            "sponsor": "DataGenome Pharma Inc.", "standard": "SDTM 3.3 + ADaM 1.3",
            "cutoff": "May 31, 2026", "total_records": 10240,
        }
    })

    # 2. Table-level chunks
    for t in CLINICAL_TABLES:
        col_names = " ".join(c["name"] for c in t["columns"])
        pks = " ".join(c["name"] for c in t["columns"] if c["isPrimary"])
        fks = " ".join(c["name"] for c in t["columns"] if c.get("isForeign"))
        chunks.append({
            "id": f"table_{t['name']}", "category": "schema",
            "title": f"Table {t['name']} — {t['label']} ({t['standard']})",
            "text": (
                f"Table {t['name']} label {t['label']} standard {t['standard']} "
                f"rows {t['rowCount']} {t['description']} "
                f"columns {col_names} primary key {pks} foreign key {fks}"
            ),
            "data": t
        })

    # 3. Column-level chunks
    for t in CLINICAL_TABLES:
        for c in t["columns"]:
            fk = (f"foreign key references {c.get('foreignTable','')} {c.get('foreignColumn','')}"
                  if c.get("isForeign") else "")
            samples = " ".join(str(s) for s in c["sampleData"][:3])
            chunks.append({
                "id": f"col_{t['name']}_{c['name']}", "category": "column",
                "title": f"Column {t['name']}.{c['name']} ({c['type']})",
                "text": (
                    f"Column {c['name']} table {t['name']} {t['label']} "
                    f"type {c['type']} nullable {c['nullable']} "
                    f"primary key {c['isPrimary']} {fk} "
                    f"standard {c['cdiscStandard']} "
                    f"description {c['description']} samples {samples}"
                ),
                "data": {**c, "table_name": t["name"], "table_label": t["label"],
                         "table_standard": t["standard"]}
            })

    # 4. DQ issue chunks
    for i in MOCK_DQ_ISSUES:
        chunks.append({
            "id": f"dq_{i['id']}", "category": "dq",
            "title": f"DQ Issue {i['id'].upper()} — {i['tableId'].upper()}.{i['columnName']} ({i['severity']})",
            "text": (
                f"data quality issue {i['id']} table {i['tableId'].upper()} "
                f"column {i['columnName']} type {i['issueType']} "
                f"severity {i['severity']} count {i['count']} "
                f"percent {i['percentage']} "
                f"description {i['description']} "
                f"remediation {i['remediation']}"
            ),
            "data": i
        })

    # 5. DQ Summary
    crit_ids  = [i['id'] for i in MOCK_DQ_ISSUES if i['severity'] == 'Critical']
    warn_ids  = [i['id'] for i in MOCK_DQ_ISSUES if i['severity'] == 'Warning']
    affected  = sum(i['count'] for i in MOCK_DQ_ISSUES)
    chunks.append({
        "id": "dq_summary", "category": "dq",
        "title": "DQ Audit Summary — All Issues",
        "text": (
            f"data quality audit summary total issues {len(MOCK_DQ_ISSUES)} "
            f"critical {len(crit_ids)} warning {len(warn_ids)} "
            f"affected rows {affected} missing null negative value duplicate "
            f"primary key future date constraint violation "
            f"database lock blocking submission remediation "
            f"DM ARMCD USUBJID VS VSSTRESN AE AESTDTC AEENDTC LB LBSTRESN"
        ),
        "data": {
            "total": len(MOCK_DQ_ISSUES), "critical": len(crit_ids),
            "warnings": len(warn_ids), "affected_rows": affected
        }
    })

    # 6. Glossary chunks
    for g in GLOSSARY:
        chunks.append({
            "id": f"gloss_{g['term'].lower().replace('+','plus').replace(' ','_')}",
            "category": "glossary",
            "title": f"Glossary: {g['term']} ({g['standard']})",
            "text": (
                f"term {g['term']} standard {g['standard']} "
                f"category {g['category']} "
                f"definition {g['definition']} example {g['example']}"
                + (f" code {g['code']}" if g.get("code") else "")
            ),
            "data": g
        })

    # 7. ER relationship chunks
    rels = [
        ("DM","AE","1:Many","USUBJID STUDYID",
         "one subject many adverse events AE USUBJID foreign key to DM join"),
        ("DM","VS","1:Many","USUBJID STUDYID",
         "one subject many vital signs VS USUBJID foreign key to DM join"),
        ("DM","LB","1:Many","USUBJID STUDYID",
         "one subject many lab results LB USUBJID foreign key to DM join"),
        ("DM","ADSL","1:1","USUBJID",
         "one subject one analysis record ADSL ADaM derived from SDTM DM subject level"),
    ]
    for f, to, rt, k, d in rels:
        chunks.append({
            "id": f"er_{f}_{to}", "category": "er",
            "title": f"ER Relationship: {f} → {to} ({rt})",
            "text": (
                f"entity relationship {f} to {to} type {rt} "
                f"join key {k} {d} "
                f"all SDTM domains link to DM via USUBJID central subject identifier"
            ),
            "data": {"from": f, "to": to, "rel": rt, "key": k, "desc": d}
        })

    # 8. Regulatory
    chunks.append({
        "id": "regulatory", "category": "regulatory",
        "title": "Regulatory Submission — FDA/EMA/PMDA Requirements",
        "text": (
            "FDA PMDA EMA regulatory submission SDTM ADaM define xml "
            "Pinnacle 21 validation GCP ICH E6 R2 ALCOA+ database lock "
            "critical issues blocking DM ARMCD missing USUBJID duplicate "
            "VS VSSTRESN negative value submission readiness compliance check "
            "dataset inventory validated dossier synopsis"
        ),
        "data": {}
    })

    # 9. Analysis populations
    chunks.append({
        "id": "analysis_populations", "category": "analysis",
        "title": "Analysis Populations — ITT, Safety, Per-Protocol",
        "text": (
            "intent to treat ITT population all randomized subjects ITTFL Y ADSL "
            "safety population SAF subjects received at least one dose SAFFL Y ADSL "
            "per protocol population no major deviations "
            "treatment arm active ACT placebo PLAC ARMCD ACTARMCD DM "
            "TRT01P planned treatment TRT01A actual treatment ADSL "
            "age group AGEGR1 under 65 over 65 stratified analysis"
        ),
        "data": {}
    })

    # 10. Adverse events summary
    chunks.append({
        "id": "ae_summary", "category": "ae",
        "title": "Adverse Events Summary — AE Domain",
        "text": (
            "adverse events AE total 1140 serious AESER Y 47 "
            "severity AESEV mild 624 moderate 398 severe 118 "
            "most common upper respiratory tract infection URTI headache "
            "MedDRA SOC nervous system infections body system "
            "AETERM verbatim AEDECOD preferred term AEBODSYS organ class "
            "AESTDTC start date AEENDTC end date resolution "
            "DQ issue future date chronology violation"
        ),
        "data": {
            "total": 1140, "serious": 47,
            "mild": 624, "moderate": 398, "severe": 118,
            "most_common": "Upper Respiratory Tract Infection"
        }
    })

    return chunks


# ─────────────────────────── BM25 Index ──────────────────────────────────────
class ClinicalRAG:
    """Zero-cost BM25 RAG. No API · No GPU · No paid services."""

    def __init__(self):
        self.chunks = _build_chunks()
        self._index = BM25Okapi([
            _tok(c["text"] + " " + c["title"]) for c in self.chunks
        ])

    def retrieve(self, query: str, top_k: int = 6) -> list:
        tokens = _tok(query)
        if not tokens:
            return []
        scores = self._index.get_scores(tokens)
        ranked = sorted(
            [(i, s) for i, s in enumerate(scores) if s > 0],
            key=lambda x: x[1], reverse=True
        )[:top_k]
        results = []
        for idx, score in ranked:
            c = dict(self.chunks[idx])
            c["score"] = round(float(score), 3)
            results.append(c)
        return results

    def get_top_categories(self, query: str, top_k: int = 6) -> list:
        return [c["category"] for c in self.retrieve(query, top_k)]

    def stats(self) -> dict:
        by_cat = {}
        for c in self.chunks:
            by_cat[c["category"]] = by_cat.get(c["category"], 0) + 1
        return {"total": len(self.chunks), "by_category": by_cat}
