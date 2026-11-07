"""File import and variable-type inference. Never drops rows; raw strings are kept for CSV."""
from __future__ import annotations

import io
import json
import re
import warnings

import numpy as np
import pandas as pd

KINDS = ["numeric", "categorical", "boolean", "datetime", "text", "identifier"]
SUPPORTED = ("csv", "tsv", "txt", "xlsx", "xls", "json", "parquet")
MAX_BYTES = 200 * 1024 * 1024
TRUE = {"true", "t", "yes", "y", "1"}
FALSE = {"false", "f", "no", "n", "0"}
_ID1 = re.compile(r"(^|[^a-z])(id|uuid|guid)($|[^a-z])", re.I)
_ID2 = re.compile(r"[a-z]Id$")
_DATEISH = r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{1,2}:\d{2}"


class LoadError(Exception):
    """Raised with a user-readable message."""


def excel_sheets(data: bytes) -> list[str]:
    try:
        return pd.ExcelFile(io.BytesIO(data)).sheet_names
    except ImportError:
        raise LoadError("Reading Excel files needs the 'openpyxl' package (pip install openpyxl).")
    except Exception as e:
        raise LoadError(f"Could not read the Excel file: {e}")


def load_file(name: str, data: bytes, encoding: str = "utf-8", sheet: str | None = None) -> pd.DataFrame:
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in SUPPORTED:
        raise LoadError(f"Unsupported file type '.{ext}'. Use CSV, TSV, Excel, JSON or Parquet.")
    if not data:
        raise LoadError("The file is empty.")
    if len(data) > MAX_BYTES:
        raise LoadError(f"The file is larger than {MAX_BYTES // 2**20} MB.")
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
        sniff = ext == "txt"
        sep = None if sniff else ("\t" if ext == "tsv" else ",")
        df = pd.read_csv(buf, dtype=str, encoding=encoding, sep=sep, engine="python" if sniff else "c")
        if df.shape[1] == 1 and re.search(r"[;\t|]", str(df.columns[0])):  # wrong delimiter guess
            df = pd.read_csv(io.BytesIO(data), dtype=str, encoding=encoding, sep=None, engine="python")
        return df
    if ext in ("xlsx", "xls"):
        return pd.read_excel(buf, sheet_name=sheet or 0)
    if ext == "json":
        obj = json.loads(data.decode(encoding))
        if isinstance(obj, dict):
            lists = [v for v in obj.values() if isinstance(v, list)]
            obj = lists[0] if len(lists) == 1 else [obj]
        return _flatten(pd.json_normalize(obj))
    return _flatten(pd.read_parquet(buf))


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
