"""
data_loader.py — Universal data ingestion layer
Supports: CSV, TSV, TXT (delimited), Excel (.xlsx/.xls), JSON, SQLite (.db/.sqlite)
Also supports live DB connections: SQLite URI, PostgreSQL, MySQL (via SQLAlchemy)
Returns unified DatasetInfo objects for the analysis engine.
"""
import io, re, json, sqlite3, csv
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

# ── DatasetInfo ───────────────────────────────────────────────────────────────
@dataclass
class ColumnInfo:
    name:        str
    dtype:       str          # pandas dtype string
    nullable:    bool
    unique_count: int
    sample:      list
    # Numeric stats (None for categoricals)
    min_val:     Optional[float] = None
    max_val:     Optional[float] = None
    mean_val:    Optional[float] = None
    median_val:  Optional[float] = None
    std_val:     Optional[float] = None
    missing:     int = 0
    missing_pct: float = 0.0
    # Categorical stats
    top_values:  dict = field(default_factory=dict)  # value → count
    is_numeric:  bool = False
    is_datetime: bool = False

@dataclass
class DatasetInfo:
    name:         str           # table or file name
    source:       str           # "csv", "excel", "sqlite", "postgresql", "mysql", "json", "txt"
    row_count:    int
    col_count:    int
    columns:      list          # list[ColumnInfo]
    df:           object        # actual pandas DataFrame (in memory)
    file_size_kb: float = 0.0
    sheet_name:   str = ""      # for Excel
    db_path:      str = ""      # for SQLite/DB
    description:  str = ""      # auto-generated summary

# ── Profiler ──────────────────────────────────────────────────────────────────
def _profile_df(df: pd.DataFrame, name: str, source: str,
                file_size_kb: float = 0.0, sheet_name: str = "",
                db_path: str = "") -> DatasetInfo:
    """Build a full DatasetInfo from any DataFrame."""
    columns = []
    for col in df.columns:
        series   = df[col]
        missing  = int(series.isna().sum())
        miss_pct = round(missing / len(series) * 100, 2) if len(series) > 0 else 0.0
        uniq     = int(series.nunique())
        sample   = [str(v) for v in series.dropna().head(4).tolist()]

        # Try numeric conversion
        num = pd.to_numeric(series, errors="coerce")
        is_num = num.notna().sum() > len(series) * 0.5

        # Top values for categoricals
        top_vals = {}
        if not is_num and uniq <= 200:
            vc = series.value_counts().head(8)
            top_vals = {str(k): int(v) for k, v in vc.items()}

        ci = ColumnInfo(
            name         = str(col),
            dtype        = str(series.dtype),
            nullable     = missing > 0,
            unique_count = uniq,
            sample       = sample,
            missing      = missing,
            missing_pct  = miss_pct,
            top_values   = top_vals,
            is_numeric   = is_num,
        )
        if is_num:
            ci.min_val    = round(float(num.min()), 4)
            ci.max_val    = round(float(num.max()), 4)
            ci.mean_val   = round(float(num.mean()), 4)
            ci.median_val = round(float(num.median()), 4)
            ci.std_val    = round(float(num.std()), 4)
        columns.append(ci)

    # Auto description
    num_cols  = sum(1 for c in columns if c.is_numeric)
    cat_cols  = len(columns) - num_cols
    miss_cols = sum(1 for c in columns if c.missing > 0)
    desc = (f"{len(df):,} rows × {len(columns)} columns | "
            f"{num_cols} numeric, {cat_cols} categorical | "
            f"{miss_cols} columns with missing values")

    return DatasetInfo(
        name         = name,
        source       = source,
        row_count    = len(df),
        col_count    = len(columns),
        columns      = columns,
        df           = df,
        file_size_kb = file_size_kb,
        sheet_name   = sheet_name,
        db_path      = db_path,
        description  = desc,
    )


# ── File loaders ──────────────────────────────────────────────────────────────
def load_csv(file_bytes: bytes, filename: str) -> list:
    """Load CSV or TSV. Returns list[DatasetInfo]."""
    size_kb = len(file_bytes) / 1024
    # Detect delimiter
    sample  = file_bytes[:4096].decode("utf-8", errors="replace")
    dialect = csv.Sniffer().sniff(sample, delimiters=",\t|;")
    delim   = dialect.delimiter if dialect else ","
    df = pd.read_csv(io.BytesIO(file_bytes), delimiter=delim,
                     low_memory=False, on_bad_lines="skip")
    name = re.sub(r"\.(csv|tsv|txt)$", "", filename, flags=re.I)
    return [_profile_df(df, name, "csv", size_kb)]


def load_excel(file_bytes: bytes, filename: str) -> list:
    """Load all sheets from Excel. Returns list[DatasetInfo] (one per sheet)."""
    size_kb = len(file_bytes) / 1024
    xl  = pd.ExcelFile(io.BytesIO(file_bytes))
    out = []
    for sheet in xl.sheet_names:
        df   = xl.parse(sheet)
        name = re.sub(r"\.(xlsx|xls)$", "", filename, flags=re.I)
        tname = f"{name}.{sheet}" if len(xl.sheet_names) > 1 else name
        out.append(_profile_df(df, tname, "excel", size_kb, sheet_name=sheet))
    return out


def load_json(file_bytes: bytes, filename: str) -> list:
    """Load JSON array or records. Returns list[DatasetInfo]."""
    size_kb = len(file_bytes) / 1024
    data = json.loads(file_bytes.decode("utf-8", errors="replace"))
    if isinstance(data, list):
        df = pd.json_normalize(data)
    elif isinstance(data, dict):
        # Try orient=records sub-key
        for k, v in data.items():
            if isinstance(v, list):
                df = pd.json_normalize(v)
                break
        else:
            df = pd.DataFrame([data])
    else:
        df = pd.DataFrame([{"value": data}])
    name = re.sub(r"\.json$", "", filename, flags=re.I)
    return [_profile_df(df, name, "json", size_kb)]


def load_txt(file_bytes: bytes, filename: str) -> list:
    """Load delimited TXT. Returns list[DatasetInfo]."""
    return load_csv(file_bytes, filename)  # sniffer handles it


def load_sqlite_bytes(file_bytes: bytes, filename: str) -> list:
    """Load SQLite file uploaded as bytes. Returns list[DatasetInfo] (one per table)."""
    import tempfile, os
    size_kb = len(file_bytes) / 1024
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        f.write(file_bytes)
        tmp_path = f.name
    try:
        results = load_sqlite_path(tmp_path, filename, size_kb)
    finally:
        os.unlink(tmp_path)
    return results


def load_sqlite_path(db_path: str, label: str = "", size_kb: float = 0.0) -> list:
    """Load all tables from a SQLite file path."""
    conn   = sqlite3.connect(db_path)
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
    out    = []
    for tname in tables["name"].tolist():
        df = pd.read_sql(f"SELECT * FROM [{tname}]", conn)
        out.append(_profile_df(df, tname, "sqlite", size_kb, db_path=db_path))
    conn.close()
    return out


def load_db_uri(uri: str, tables: list = None) -> list:
    """
    Connect to PostgreSQL / MySQL / SQLite via SQLAlchemy URI.
    URI format:
      postgresql://user:pass@host:5432/dbname
      mysql+pymysql://user:pass@host:3306/dbname
      sqlite:///path/to/file.db
    """
    try:
        from sqlalchemy import create_engine, inspect
    except ImportError:
        raise ImportError("sqlalchemy not installed")

    engine  = create_engine(uri, connect_args={"connect_timeout": 10})
    insp    = inspect(engine)
    all_tables = insp.get_table_names()

    target = tables if tables else all_tables
    out    = []
    with engine.connect() as conn:
        for tname in target:
            df = pd.read_sql_table(tname, conn)
            out.append(_profile_df(df, tname, uri.split("://")[0], db_path=uri))
    return out


# ── Dispatch ──────────────────────────────────────────────────────────────────
def load_uploaded_file(file_obj) -> list:
    """
    Takes a Streamlit UploadedFile object.
    Returns list[DatasetInfo].
    """
    name  = file_obj.name.lower()
    data  = file_obj.read()

    if name.endswith(".csv"):
        return load_csv(data, file_obj.name)
    elif name.endswith((".xlsx", ".xls")):
        return load_excel(data, file_obj.name)
    elif name.endswith(".json"):
        return load_json(data, file_obj.name)
    elif name.endswith((".txt", ".tsv")):
        return load_txt(data, file_obj.name)
    elif name.endswith((".db", ".sqlite", ".sqlite3")):
        return load_sqlite_bytes(data, file_obj.name)
    else:
        # Try CSV as fallback
        try:
            return load_csv(data, file_obj.name)
        except Exception:
            raise ValueError(f"Unsupported file type: {file_obj.name}")
