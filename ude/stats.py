"""Descriptive statistics and relationship tests. Every test validates its inputs and returns
{"ok": False, "reason": ...} instead of a misleading number."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

MIN_N = 10        # minimum complete pairs for any test
MIN_GROUP = 3     # minimum observations per group
MAX_LEVELS = 30   # more categories than this is not a readable comparison


def _bad(reason):
    return {"ok": False, "reason": reason}


# ------------------------------------------------------------ descriptive
def describe_numeric(df, cols) -> pd.DataFrame:
    rows = []
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").astype(float).dropna()
        n = len(x)
        q = x.quantile([0.25, 0.75]) if n else pd.Series([np.nan, np.nan], index=[0.25, 0.75])
        rows.append(dict(variable=c, count=n, missing_pct=round(100 * (1 - n / len(df)), 2) if len(df) else 0.0,
                         mean=x.mean() if n else np.nan, median=x.median() if n else np.nan,
                         std=x.std() if n > 1 else np.nan, variance=x.var() if n > 1 else np.nan,
                         min=x.min() if n else np.nan, q1=q[0.25], q3=q[0.75], max=x.max() if n else np.nan,
                         skew=x.skew() if n >= 3 else np.nan))
    return pd.DataFrame(rows)


def describe_categorical(df, col, top=10) -> tuple[dict, pd.DataFrame]:
    s = df[col].dropna().astype(str)
    vc = s.value_counts()
    t = pd.DataFrame({"value": vc.index[:top], "count": vc.values[:top],
                      "proportion": (vc.values[:top] / len(s)).round(4) if len(s) else []})
    return dict(n=len(s), unique=int(vc.size)), t


def describe_datetime(df, col) -> tuple[dict, pd.DataFrame]:
    s = df[col].dropna().sort_values()
    if s.empty:
        return dict(n=0), pd.DataFrame(columns=["period", "count"])
    span = (s.iloc[-1] - s.iloc[0]).days
    gaps = s.diff().dropna()
    info = dict(n=len(s), start=str(s.iloc[0]), end=str(s.iloc[-1]), span_days=span,
                median_gap=str(gaps.median()) if len(gaps) else "n/a")
    freq = "M" if span > 60 else "D"
    pc = s.dt.to_period(freq).value_counts().sort_index()
    return info, pd.DataFrame({"period": pc.index.astype(str), "count": pc.values})


def describe_text(df, col) -> tuple[dict, pd.DataFrame]:
    s = df[col].dropna().astype(str)
    ln = s.str.len()
    info = dict(n=len(s), unique=int(s.nunique()),
                len_mean=round(ln.mean(), 1) if len(s) else np.nan, len_median=ln.median() if len(s) else np.nan,
                len_min=ln.min() if len(s) else np.nan, len_max=ln.max() if len(s) else np.nan,
                words_mean=round(s.str.split().str.len().mean(), 1) if len(s) else np.nan)
    vc = s.value_counts().head(10)
    return info, pd.DataFrame({"value": vc.index.str.slice(0, 60), "count": vc.values})


def summary_table(df, kinds) -> pd.DataFrame:
    """One row per variable, suitable for export."""
    num = describe_numeric(df, [c for c in df if kinds.get(c) == "numeric"]).set_index("variable")
    rows = []
    for c in df.columns:
        s = df[c].dropna()
        r = dict(variable=c, kind=kinds.get(c), count=len(s),
                 missing_pct=round(100 * df[c].isna().mean(), 2) if len(df) else 0.0, unique=int(s.nunique()))
        if c in num.index:
            r.update({k: num.loc[c, k] for k in ["mean", "median", "std", "min", "max", "skew"]})
        elif kinds.get(c) in ("categorical", "boolean") and len(s):
            vc = s.astype(str).value_counts()
            r.update(top_value=vc.index[0], top_share=round(vc.iloc[0] / len(s), 4))
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------ relationships
def bh_adjust(p) -> np.ndarray:
    """Benjamini-Hochberg false-discovery-rate adjusted p-values."""
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    adj = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1)
    return out


def corr_pair(df, x, y, method="spearman") -> dict:
    if x == y:
        return _bad("Choose two different variables.")
    d = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna().astype(float)
    n = len(d)
    if n < MIN_N:
        return _bad(f"Only {n} rows have both values; at least {MIN_N} are needed.")
    if d[x].nunique() < 2 or d[y].nunique() < 2:
        return _bad("One of the variables is constant in the rows used.")
    fn = stats.spearmanr if method == "spearman" else stats.pearsonr
    r, p = fn(d[x], d[y])
    return dict(ok=True, method=method, r=float(r), p=float(p), n=n)


def corr_matrix(df, cols, method="spearman") -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df[cols].apply(pd.to_numeric, errors="coerce").astype(float)
    n = d.notna().astype(int)
    return d.corr(method=method, min_periods=MIN_N), n.T.dot(n)


def group_compare(df, cat, num) -> dict:
    d = df[[cat, num]].dropna()
    d = d.assign(**{cat: d[cat].astype(str), num: pd.to_numeric(d[num], errors="coerce")}).dropna()
    sizes = d.groupby(cat)[num].size()
    keep = sizes[sizes >= MIN_GROUP].index
    if len(keep) < 2:
        return _bad(f"Fewer than 2 groups have at least {MIN_GROUP} observations.")
    if len(keep) > MAX_LEVELS:
        return _bad(f"{len(keep)} groups is too many to compare (limit {MAX_LEVELS}).")
    d = d[d[cat].isin(keep)]
    if d[num].nunique() < 2:
        return _bad("The numeric variable is constant in the rows used.")
    n, k = len(d), len(keep)
    H, p = stats.kruskal(*[g.to_numpy() for _, g in d.groupby(cat)[num]])
    eps2 = float(max(0.0, (H - k + 1) / (n - k)))
    table = d.groupby(cat)[num].agg(n="size", median="median", mean="mean").reset_index()
    return dict(ok=True, method="Kruskal-Wallis", H=float(H), p=float(p), eps2=eps2, n=n, groups=k,
                dropped_groups=int((sizes < MIN_GROUP).sum()), table=table.sort_values("median", ascending=False))


def cat_assoc(df, a, b) -> dict:
    if a == b:
        return _bad("Choose two different variables.")
    d = df[[a, b]].dropna().astype(str)
    n = len(d)
    if n < 2 * MIN_N:
        return _bad(f"Only {n} rows have both values; at least {2 * MIN_N} are needed.")
    ct = pd.crosstab(d[a], d[b])
    if min(ct.shape) < 2:
        return _bad("Each variable needs at least 2 categories in the rows used.")
    if max(ct.shape) > MAX_LEVELS:
        return _bad(f"More than {MAX_LEVELS} categories; the table would be unreadable and sparse.")
    chi2, p, dof, exp = stats.chi2_contingency(ct, correction=False)
    if (exp < 1).any() or (exp >= 5).mean() < 0.8:
        return _bad("Expected cell counts are too small for a chi-square test (need >=80% of cells with >=5, none <1).")
    v = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))
    return dict(ok=True, method="Chi-square test of independence", chi2=float(chi2), dof=int(dof), p=float(p),
                cramers_v=v, n=n, table=ct)


def trend(df, t, y) -> dict:
    d = df[[t, y]].dropna()
    d = d.assign(**{y: pd.to_numeric(d[y], errors="coerce")}).dropna()
    n = len(d)
    if n < MIN_N:
        return _bad(f"Only {n} rows have both a date and a value; at least {MIN_N} are needed.")
    tt = (d[t] - d[t].min()).dt.total_seconds()
    if tt.nunique() < 3 or d[y].nunique() < 2:
        return _bad("Not enough distinct dates or values to assess a trend.")
    r, p = stats.spearmanr(tt, d[y])
    return dict(ok=True, method="Spearman correlation with time", r=float(r), p=float(p), n=n)


def effect_label(kind, v) -> str:
    cuts = {"corr": (0.3, 0.5, 0.8), "trend": (0.3, 0.5, 0.8), "assoc": (0.2, 0.4, 0.6), "group": (0.08, 0.26, 0.5)}[kind]
    names = ["negligible-to-weak", "moderate", "strong", "very strong"]
    return names[sum(v >= c for c in cuts)]
