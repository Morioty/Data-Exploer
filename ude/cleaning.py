"""Reversible cleaning: the original is never touched; the current table is always replay(original, ops)."""
from __future__ import annotations

import json
import re

import pandas as pd

from .loading import apply_types

OPS = ["drop_duplicates", "fill_missing", "rename", "convert_type", "filter_rows", "select_columns", "standardize_text"]
FILTER_OPS = ["==", "!=", ">", ">=", "<", "<=", "contains", "is missing", "is not missing"]


def _need(df, cols):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"Unknown column(s): {', '.join(miss)}")


def _cast(s: pd.Series, v):
    if pd.api.types.is_numeric_dtype(s.dtype) and not pd.api.types.is_bool_dtype(s.dtype):
        return float(v)
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return pd.Timestamp(v)
    if pd.api.types.is_bool_dtype(s.dtype):
        return str(v).strip().lower() in ("true", "1", "yes")
    return str(v)


def apply_op(df: pd.DataFrame, op: dict) -> pd.DataFrame:
    t, p = op["type"], op.get("params", {})
    if t == "drop_duplicates":
        return df.drop_duplicates()
    if t == "select_columns":
        _need(df, p["columns"])
        if not p["columns"]:
            raise ValueError("Keep at least one column.")
        return df[list(p["columns"])]
    if t == "rename":
        _need(df, p["mapping"])
        out = df.rename(columns=p["mapping"])
        if out.columns.duplicated().any() or any(not str(c).strip() for c in out.columns):
            raise ValueError("New names must be non-empty and unique.")
        return out
    if t == "convert_type":
        _need(df, [p["column"]])
        return apply_types(df, {p["column"]: p["to"]})
    if t == "standardize_text":
        _need(df, p["columns"])
        out = df.copy()
        for c in p["columns"]:
            s = out[c].astype("string")
            if p.get("strip", True):
                s = s.str.strip()
            if p.get("collapse", True):
                s = s.str.replace(r"\s+", " ", regex=True)
            if p.get("lower"):
                s = s.str.lower()
            out[c] = s
        return out
    if t == "fill_missing":
        cols, how = p["columns"], p["strategy"]
        _need(df, cols)
        if how == "drop_rows":
            return df.dropna(subset=cols)
        out = df.copy()
        for c in cols:
            s = out[c]
            if how in ("mean", "median"):
                if not pd.api.types.is_numeric_dtype(s.dtype):
                    raise ValueError(f"'{c}' is not numeric; {how} is not defined.")
                v = getattr(s, how)()
            elif how == "mode":
                m = s.mode(dropna=True)
                if m.empty:
                    continue
                v = m.iloc[0]
            elif how == "constant":
                v = _cast(s, p["value"])
            else:
                raise ValueError(f"Unknown strategy {how}")
            out[c] = s.fillna(v)
        return out
    if t == "filter_rows":
        c, o, v = p["column"], p["op"], p.get("value")
        _need(df, [c])
        s = df[c]
        if o == "is missing":
            m = s.isna()
        elif o == "is not missing":
            m = s.notna()
        elif o == "contains":
            m = s.astype("string").str.contains(str(v), case=False, regex=False).fillna(False)
        else:
            x = _cast(s, v)
            m = {"==": s == x, "!=": s != x, ">": s > x, ">=": s >= x, "<": s < x, "<=": s <= x}[o].fillna(False)
        return df[m.astype(bool)]
    raise ValueError(f"Unknown operation {t}")


def replay(original: pd.DataFrame, ops: list[dict]) -> pd.DataFrame:
    df = original
    for op in ops:
        df = apply_op(df, op)
    return df


def add_op(original, ops, op, desc) -> list[dict]:
    """Return ops + [op] after validating it; raises ValueError and leaves ops unchanged on failure."""
    before = replay(original, ops)
    try:
        after = apply_op(before, op)
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(str(e))
    return ops + [dict(op, desc=desc, rows_before=len(before), rows_after=len(after),
                       cols_before=before.shape[1], cols_after=after.shape[1])]


def log_json(source: str, original: pd.DataFrame, ops: list[dict]) -> str:
    return json.dumps(dict(source=source, original_shape=list(original.shape),
                           operations=[{k: v for k, v in o.items()} for o in ops]), indent=2, default=str)
