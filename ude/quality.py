"""Dataset overview and data-quality checks. Reports only; never modifies data."""
from __future__ import annotations

import numpy as np
import pandas as pd

OUTLIER_NOTE = (
    "IQR rule flags values beyond 1.5×IQR from the quartiles; it ignores skew, so skewed data over-flags. "
    "Z-score flags |z|>3; it assumes roughly normal data and is itself distorted by the outliers it looks for. "
    "A flagged value is unusual, not necessarily an error. Nothing is removed."
)


def overview(df: pd.DataFrame, kinds: dict) -> dict:
    cells = df.shape[0] * df.shape[1]
    return dict(
        rows=len(df), cols=df.shape[1],
        memory_mb=round(df.memory_usage(deep=True).sum() / 2**20, 2),
        missing_pct=round(100 * df.isna().sum().sum() / cells, 2) if cells else 0.0,
        duplicates=int(df.duplicated().sum()) if len(df) else 0,
        kind_counts=pd.Series(kinds).value_counts().to_dict(),
    )


def constant_columns(df) -> list[str]:
    return [c for c in df.columns if df[c].nunique(dropna=True) <= 1 and df[c].notna().any()]


def high_cardinality(df, kinds, limit=50) -> list[str]:
    return [c for c in df.columns if kinds.get(c) == "categorical" and df[c].nunique() > limit]


def mixed_type_columns(raw: pd.DataFrame) -> dict[str, float]:
    """Text columns where 5-95% of values parse as numbers (share returned)."""
    out = {}
    for c in raw.columns:
        if raw[c].dtype == object or isinstance(raw[c].dtype, pd.StringDtype):
            s = raw[c].dropna().astype(str).str.strip()
            if len(s):
                r = pd.to_numeric(s, errors="coerce").notna().mean()
                if 0.05 < r < 0.95:
                    out[c] = float(r)
    return out


def coercion_losses(raw: pd.DataFrame, typed: pd.DataFrame) -> dict[str, int]:
    """Values present in the raw file that became missing after type conversion."""
    out = {}
    for c in typed.columns:
        if c in raw:
            n = int((raw[c].notna() & typed[c].isna()).sum())
            if n:
                out[c] = n
    return out


def inconsistent_categories(df, kinds, max_show=5) -> dict[str, list[list[str]]]:
    """Category spellings that differ only by case/whitespace."""
    out = {}
    for c in df.columns:
        if kinds.get(c) != "categorical":
            continue
        vals = df[c].dropna().astype(str).unique()
        if len(vals) > 500:
            continue
        groups = {}
        for v in vals:
            groups.setdefault(" ".join(v.lower().split()), []).append(v)
        bad = [g for g in groups.values() if len(g) > 1]
        if bad:
            out[c] = bad[:max_show]
    return out


def missing_by_column(df) -> pd.DataFrame:
    m = df.isna().sum()
    return pd.DataFrame({"column": m.index, "missing": m.values,
                         "missing_%": (100 * m.values / max(len(df), 1)).round(2)}).sort_values("missing", ascending=False)


def missing_patterns(df, top=10) -> pd.DataFrame:
    """Most common combinations of columns missing together."""
    cols = [c for c in df.columns if df[c].isna().any()]
    if not cols:
        return pd.DataFrame(columns=["columns_missing_together", "rows", "rows_%"])
    key = df[cols].isna().apply(lambda r: ", ".join(c for c, v in zip(cols, r) if v) or "(none)", axis=1)
    vc = key.value_counts().head(top)
    return pd.DataFrame({"columns_missing_together": vc.index, "rows": vc.values,
                         "rows_%": (100 * vc.values / len(df)).round(2)})


def outliers(s: pd.Series, z=3.0, k=1.5) -> dict | None:
    x = pd.to_numeric(s, errors="coerce").dropna().astype(float)
    n = len(x)
    if n < 8 or x.nunique() < 2:
        return None
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    sd = x.std(ddof=1)
    zc = int(((x - x.mean()).abs() / sd > z).sum()) if sd > 0 else 0
    ic = int(((x < lo) | (x > hi)).sum())
    return dict(n=n, iqr_lower=lo, iqr_upper=hi, iqr_count=ic, iqr_pct=round(100 * ic / n, 2),
                z_count=zc, z_pct=round(100 * zc / n, 2), iqr_is_zero=bool(iqr == 0))


def data_dictionary(df, kinds, mixed=None) -> pd.DataFrame:
    mixed = mixed or {}
    const = set(constant_columns(df))
    hc = set(high_cardinality(df, kinds))
    rows = []
    for c in df.columns:
        s = df[c]
        flags = []
        if s.isna().all():
            flags.append("all missing")
        if c in const:
            flags.append("constant")
        if c in hc:
            flags.append("high cardinality")
        if c in mixed:
            flags.append("mixed numbers/text")
        nn = s.dropna()
        rows.append(dict(column=c, kind=kinds.get(c), dtype=str(s.dtype), missing=int(s.isna().sum()),
                         missing_pct=round(100 * s.isna().mean(), 2) if len(s) else 0.0,
                         unique=int(nn.nunique()), example=str(nn.iloc[0])[:40] if len(nn) else "",
                         flags="; ".join(flags)))
    return pd.DataFrame(rows)


def quality_warnings(df, kinds, raw=None, typed=None) -> list[dict]:
    """List of {level, text}. level: warning|info."""
    w = []
    add = lambda lvl, t: w.append(dict(level=lvl, text=t))
    n = len(df)
    if n < 30:
        add("warning", f"Only {n} rows: many statistical tests will be skipped or unreliable.")
    ov = overview(df, kinds)
    if ov["duplicates"]:
        add("warning", f"{ov['duplicates']} exact duplicate rows ({100 * ov['duplicates'] / n:.1f}%). They are kept until you remove them in Export.")
    for c in df.columns:
        m = df[c].isna().mean() if n else 0
        if m == 1:
            add("warning", f"'{c}' is entirely missing.")
        elif m >= 0.2:
            add("warning", f"'{c}' is {100 * m:.0f}% missing.")
    for c in constant_columns(df):
        add("info", f"'{c}' has a single value; it cannot be compared with anything.")
    for c in high_cardinality(df, kinds):
        add("info", f"'{c}' has {df[c].nunique()} distinct categories; group comparisons will skip it.")
    for c, g in inconsistent_categories(df, kinds).items():
        add("warning", f"'{c}' has spellings that differ only by case/spaces, e.g. {g[0]}.")
    if raw is not None:
        for c, r in mixed_type_columns(raw).items():
            add("warning", f"'{c}' mixes numbers ({100 * r:.0f}%) and text; check its type.")
    if raw is not None and typed is not None:
        for c, k in coercion_losses(raw, typed).items():
            add("warning", f"{k} value(s) in '{c}' could not be converted to its type and became missing.")
    for c in df.columns:
        if kinds.get(c) == "numeric":
            o = outliers(df[c])
            if o and o["iqr_pct"] >= 5:
                add("info", f"'{c}': {o['iqr_pct']}% of values fall outside the IQR fences (unusual, not necessarily wrong).")
    return w
