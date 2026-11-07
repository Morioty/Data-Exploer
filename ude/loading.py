"""File import and variable-type inference. Never drops rows; raw strings are kept for CSV."""
from __future__ import annotations

import io
import json
import re
import warnings

import numpy as np
import pandas as pd

KINDS = ["numeric", "categorical", "boolean", "datetime", "text", "identifier"]
SUPPORTED = ("csv", "tsv", "txt", "xlsx", "xls", "ods", "json", "jsonl", "ndjson", "parquet", "feather", "xml",
             "html", "htm", "md", "markdown", "dta", "db", "sqlite", "sqlite3")
EXCEL_LIKE = ("xlsx", "xls", "ods")
TABLE_PICK = ("html", "htm", "md", "markdown", "db", "sqlite", "sqlite3")
FALLBACK_ENCODINGS = ("utf-8-sig", "cp949", "latin-1")  # cp949 covers Korean CSVs from Excel
MAX_BYTES = 200 * 1024 * 1024
TRUE = {"true", "t", "yes", "y", "1"}
FALSE = {"false", "f", "no", "n", "0"}
_ID1 = re.compile(r"(^|[^a-z])(id|uuid|guid)($|[^a-z])", re.I)
_ID2 = re.compile(r"[a-z]Id$")
_DATEISH = r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{1,2}:\d{2}"


class LoadError(Exception):
    """Raised with a user-readable message."""


def ext_of(name: str) -> str:
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def excel_sheets(data: bytes, ext: str = "xlsx") -> list[str]:
    try:
        return pd.ExcelFile(io.BytesIO(data), engine="odf" if ext == "ods" else None).sheet_names
    except ImportError:
        raise LoadError("Reading this spreadsheet needs an extra package: 'openpyxl' for Excel, 'odfpy' for .ods (pip install it).")
    except Exception as e:
        raise LoadError(f"Could not read the spreadsheet: {e}")


def unwrap_gz(name: str, data: bytes) -> tuple[str, bytes]:
    """Transparently decompress .gz (size-capped against decompression bombs)."""
    import gzip
    try:
        out = gzip.GzipFile(fileobj=io.BytesIO(data)).read(MAX_BYTES + 1)
    except Exception as e:
        raise LoadError(f"Could not decompress the .gz file: {e}")
    if len(out) > MAX_BYTES:
        raise LoadError(f"The decompressed file is larger than {MAX_BYTES // 2**20} MB.")
    return name[:-3], out


def zip_members(data: bytes) -> list[str]:
    import zipfile
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except Exception as e:
        raise LoadError(f"Could not open the .zip file: {e}")
    ok = [i.filename for i in zf.infolist() if not i.is_dir() and "__MACOSX" not in i.filename
          and ext_of(i.filename) in SUPPORTED + ("gz",) and i.file_size <= MAX_BYTES]
    if not ok:
        raise LoadError("The .zip contains no readable data files (CSV, Excel, JSON, Parquet, …).")
    return ok


def zip_read(data: bytes, member: str) -> bytes:
    import zipfile
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        out = zf.open(member).read(MAX_BYTES + 1)  # read is capped, not trusted from the header
    if len(out) > MAX_BYTES:
        raise LoadError(f"'{member}' is larger than {MAX_BYTES // 2**20} MB once extracted.")
    return out


def _text(data: bytes, encoding: str) -> str:
    """Decode with the chosen encoding, then fall back (BOM-aware, then CP949, then Latin-1)."""
    tries = [encoding]
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        tries.append("utf-16")
    tries += [e for e in FALLBACK_ENCODINGS if e != encoding]
    for enc in tries:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(encoding, data, 0, 1, "no encoding worked")


def load_file(name: str, data: bytes, encoding: str = "utf-8", sheet: str | None = None) -> pd.DataFrame:
    ext = ext_of(name)
    if ext == "gz":
        name, data = unwrap_gz(name, data)
        ext = ext_of(name)
    if ext == "zip":
        raise LoadError("Choose a file inside the .zip first.")
    if not data:
        raise LoadError("The file is empty.")
    if len(data) > MAX_BYTES:
        raise LoadError(f"The file is larger than {MAX_BYTES // 2**20} MB.")
    if data[:4] == b"%PDF" or ext in ("pdf", "png", "jpg", "jpeg", "gif", "docx", "pptx"):
        raise LoadError("This is a document or image, not a data table. Export the table to CSV/Excel first.")
    if ext not in SUPPORTED:
        if b"\x00" in data[:4096] and data[:2] not in (b"\xff\xfe", b"\xfe\xff"):
            raise LoadError(f"Unrecognised binary file type '.{ext}'. Supported: CSV, TSV, Excel, ODS, JSON/JSONL, Parquet, "
                            "Feather, XML, HTML/Markdown tables, SQLite, Stata, and .gz/.zip of these.")
        ext = "txt"  # unknown extension but text: sniff the delimiter
    try:
        df = _read(ext, data, encoding, sheet)
    except LoadError:
        raise
    except UnicodeDecodeError:
        raise LoadError(f"Could not decode the file as '{encoding}'. Try another encoding (e.g. latin-1).")
    except ImportError as e:
        raise LoadError(f"A package needed for .{ext} files is missing: {e}")
    except pd.errors.EmptyDataError:
        raise LoadError("The file contains no data.")
    except Exception as e:
        raise LoadError(f"Could not parse the file as {ext.upper()}: {e}")
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise LoadError("The file has a header but no data rows." if df.shape[1] else "The file has no columns.")
    df.columns = _unique([str(c) for c in df.columns])
    return df.reset_index(drop=True)


def _read(ext, data, encoding, sheet):
    buf = io.BytesIO(data)
    if ext in ("csv", "tsv", "txt"):
        text = _text(data, encoding)
        sniff = ext == "txt"
        sep = None if sniff else ("\t" if ext == "tsv" else ",")
        df = pd.read_csv(io.StringIO(text), dtype=str, sep=sep, engine="python" if sniff else "c")
        if df.shape[1] == 1 and re.search(r"[;\t|]", str(df.columns[0])):  # wrong delimiter guess
            df = pd.read_csv(io.StringIO(text), dtype=str, sep=None, engine="python")
        if ext == "txt" and df.shape[1] == 1:
            raise LoadError("Could not find a delimiter (comma, tab, semicolon, pipe); this does not look like a table.")
        return df
    if ext in EXCEL_LIKE:
        return pd.read_excel(buf, sheet_name=sheet or 0, engine="odf" if ext == "ods" else None)
    if ext in ("html", "htm", "md", "markdown"):
        ts = _tables(ext, _text(data, encoding))
        if not ts:
            raise LoadError("No table with a header and at least one data row was found in this file.")
        if sheet:
            return ts[int(re.match(r"Table (\d+)", sheet).group(1)) - 1]
        return max(ts, key=len)
    if ext in ("db", "sqlite", "sqlite3"):
        con = _sqlite(data)
        names = _sqlite_tables(con)
        t = sheet or (names[0] if names else None)
        if t not in names:
            raise LoadError("The database has no tables." if not names else f"Table '{t}' not found.")
        return pd.read_sql_query('SELECT * FROM "' + t.replace('"', '""') + '"', con)
    if ext in ("json", "jsonl", "ndjson"):
        text = _text(data, encoding)
        try:
            obj = json.loads(text)
        except ValueError:  # JSON Lines: one object per line
            obj = [json.loads(l) for l in text.splitlines() if l.strip()]
        if isinstance(obj, dict):
            lists = [v for v in obj.values() if isinstance(v, list)]
            obj = lists[0] if len(lists) == 1 else [obj]
        return _flatten(pd.json_normalize(obj))
    if ext == "xml":  # etree parser: no network access, no external entities
        return pd.read_xml(io.StringIO(_text(data, encoding)), parser="etree", dtype=str)
    if ext == "feather":
        return _flatten(pd.read_feather(buf))
    if ext == "dta":
        return pd.read_stata(buf)
    return _flatten(pd.read_parquet(buf))


def _sqlite(data):
    import sqlite3
    con = sqlite3.connect(":memory:")
    con.deserialize(data)
    return con


def _sqlite_tables(con) -> list[str]:
    return [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY name")]


def _rows_to_df(rows):
    rows = [r for r in rows if any(c for c in r)]
    if len(rows) < 2:
        return None
    w = max(len(r) for r in rows)
    rows = [r + [""] * (w - len(r)) for r in rows]
    df = pd.DataFrame(rows[1:], columns=[h or f"column_{i + 1}" for i, h in enumerate(rows[0])], dtype=str)
    return df.replace("", np.nan)  # all text, so leading zeros and ids survive


def _tables(ext, text) -> list[pd.DataFrame]:
    out = []
    if ext in ("html", "htm"):
        from lxml import html as lh  # parses only; never fetches anything
        for t in lh.fromstring(text).iter("table"):
            rows = [[" ".join(c.text_content().split()) for c in tr if c.tag in ("td", "th")] for tr in t.iter("tr")]
            if (d := _rows_to_df(rows)) is not None:
                out.append(d)
        return out
    block = []
    for line in text.splitlines() + [""]:
        if "|" in line:
            block.append(line.strip())
            continue
        if len(block) >= 3 and re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?", block[1]):
            cells = lambda l: [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", l.strip("|"))]
            if (d := _rows_to_df([cells(block[0])] + [cells(l) for l in block[2:]])) is not None:
                out.append(d)
        block = []
    return out


def table_names(name: str, data: bytes, encoding: str = "utf-8") -> list[str]:
    """Labels for the tables in an HTML/Markdown file, largest first."""
    ext = ext_of(name)
    try:
        if ext in ("db", "sqlite", "sqlite3"):
            return _sqlite_tables(_sqlite(data))
        ts = _tables(ext, _text(data, encoding))
    except UnicodeDecodeError:
        raise LoadError(f"Could not decode the file as '{encoding}'.")
    except ImportError as e:
        raise LoadError(f"A package needed for HTML files is missing: {e}")
    except Exception as e:
        raise LoadError(f"Could not read tables: {e}")
    order = sorted(range(len(ts)), key=lambda i: -len(ts[i]))
    return [f"Table {i + 1} ({len(ts[i])} rows × {ts[i].shape[1]} cols)" for i in order]


def _flatten(df):
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v)
    return df


def _unique(names):
    seen, out = {}, []
    for n in names:
        k = seen.get(n, 0)
        seen[n] = k + 1
        out.append(n if k == 0 else f"{n}_{k}")
    return out


# ---------------------------------------------------------------- inference
def _is_id_name(name) -> bool:
    return bool(_ID1.search(str(name)) or _ID2.search(str(name)))


def _numeric_kind(x: pd.Series, name) -> str:
    n = len(x)
    vals = x.unique()
    if len(vals) == 2 and set(vals) <= {0, 1}:
        return "boolean"
    is_int = bool((x % 1 == 0).all())
    ratio = x.nunique() / n
    if is_int and _is_id_name(name) and ratio > 0.9:
        return "identifier"
    return "numeric"


def _kind_obj(s: pd.Series, name) -> str:
    s = s.astype(str).str.strip()
    s = s[s != ""]
    n = len(s)
    if n == 0:
        return "categorical"
    u = set(s.str.lower().unique())
    if len(u) <= 2 and u <= (TRUE | FALSE):
        return "boolean"
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() >= 0.95:
        if s[num.notna()].str.match(r"^[+-]?0\d").any():  # zip codes etc.: keep as text
            return "identifier"
        return _numeric_kind(num.dropna(), name)
    probe = s.sample(min(n, 1000), random_state=0)
    if probe.str.contains(_DATEISH, regex=True).mean() > 0.8:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ok = pd.to_datetime(probe, errors="coerce", format="mixed", utc=True).notna().mean()
        if ok >= 0.9:
            return "datetime"
    nu = s.nunique()
    words = s.str.split().str.len().mean()
    length = s.str.len().mean()
    if n >= 20 and nu / n > 0.9:
        return "identifier" if (words < 2 or _is_id_name(name)) else "text"
    if nu <= 50 or nu / n <= 0.2:
        return "categorical"
    return "text" if (length > 30 or words > 3) else "categorical"


def infer_types(df: pd.DataFrame) -> dict[str, str]:
    kinds = {}
    for c in df.columns:
        s = df[c].dropna()
        t = s.dtype
        if len(s) == 0:
            k = "categorical"
        elif pd.api.types.is_bool_dtype(t):
            k = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(t):
            k = "datetime"
        elif pd.api.types.is_numeric_dtype(t):
            k = _numeric_kind(s.astype(float), c)
        else:
            k = _kind_obj(s, c)
        kinds[c] = k
    return kinds


def _to_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s.dtype):
        return s.astype("boolean")
    if pd.api.types.is_numeric_dtype(s.dtype):
        out = (s == 1).astype("boolean")
        out[~s.isin([0, 1])] = pd.NA
        return out
    m = {**{k: True for k in TRUE}, **{k: False for k in FALSE}}
    return s.astype("string").str.strip().str.lower().map(m).astype("boolean")


def apply_types(df: pd.DataFrame, kinds: dict[str, str]) -> pd.DataFrame:
    """Return a typed copy. Unparseable values become missing (counted by quality.coercion_losses)."""
    out = df.copy()
    for c, k in kinds.items():
        if c not in out:
            continue
        s = out[c]
        if k == "numeric":
            out[c] = pd.to_numeric(s, errors="coerce")
        elif k == "datetime":
            if not pd.api.types.is_datetime64_any_dtype(s.dtype):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out[c] = pd.to_datetime(s.astype("string"), errors="coerce", format="mixed", utc=True).dt.tz_localize(None)
        elif k == "boolean":
            out[c] = _to_bool(s)
        elif not isinstance(s.dtype, pd.StringDtype):
            out[c] = s.astype("string")
    return out
