"""Plotly figures. Charts may be sampled; data used for stats/exports never is."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

MAX_POINTS = 20000


def _plain(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    for c in d.columns:
        t = d[c].dtype
        if isinstance(t, (pd.StringDtype, pd.BooleanDtype)):
            d[c] = d[c].astype(object).astype(str)
        elif pd.api.types.is_extension_array_dtype(t) and pd.api.types.is_numeric_dtype(t):
            d[c] = d[c].astype(float)
    return d


def _sample(d, n):
    if len(d) > n:
        return d.sample(n, random_state=0), f"Chart shows a random sample of {n:,} of {len(d):,} rows (statistics use all rows)."
    return d, ""


def make_fig(df: pd.DataFrame, spec: dict, sample_n: int = MAX_POINTS):
    """Return (figure | None, note). None means the chart is not possible with this data."""
    t = spec["type"]
    x, y, color = spec.get("x"), spec.get("y"), spec.get("color")
    cols = [c for c in (x, y, color) if c]
    if t not in ("missing", "corr") and not cols:
        return None, "Select a column."
    if t == "missing":
        m = (100 * df.isna().mean()).sort_values(ascending=False)
        m = m[m > 0]
        if m.empty:
            return None, "No missing values."
        return px.bar(x=m.index, y=m.values, labels=dict(x="column", y="% missing")), ""
    if t == "corr":
        cs = spec["cols"][:30]
        if len(cs) < 2:
            return None, "Need at least two numeric columns."
        c = df[cs].apply(pd.to_numeric, errors="coerce").astype(float).corr(method=spec.get("method", "spearman"), min_periods=10)
        return px.imshow(c, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", text_auto=".2f", aspect="auto"), ""
    d = _plain(df[cols].dropna())
    if d.empty:
        return None, "No rows have all selected values."
    if t == "hist":
        if spec.get("log_x") and (d[x] > 0).all():
            d, note = _sample(d, sample_n)
            f = px.histogram(d, x=x, color=color, nbins=40, log_x=True, barmode="overlay", opacity=0.7)
            return f, note
        d, note = _sample(d, sample_n)
        return px.histogram(d, x=x, color=color, nbins=40, barmode="overlay", opacity=0.7), note
    if t == "box":
        if x:
            top = d[x].value_counts().head(15).index
            d = d[d[x].isin(top)]
        d, note = _sample(d, sample_n)
        return px.box(d, x=x, y=y), note + (" Top 15 groups by size shown." if x and df[x].nunique() > 15 else "")
    if t == "bar":
        vc = d[x].astype(str).value_counts().head(20)
        return px.bar(x=vc.index, y=vc.values, labels=dict(x=x, y="count")), "Top 20 categories shown." if df[x].nunique() > 20 else ""
    if t == "scatter":
        d, note = _sample(d, sample_n)
        return px.scatter(d, x=x, y=y, color=color, opacity=0.6), note
    if t == "line":
        d = d.sort_values(x)
        d, note = _sample(d, min(sample_n, 5000))
        return px.line(d.sort_values(x), x=x, y=y, markers=len(d) < 300), note
    if t in ("line_agg", "line_agg_count"):
        span = (d[x].max() - d[x].min()).days
        freq = "M" if span > 180 else "D"
        g = d.assign(_p=d[x].dt.to_period(freq).dt.to_timestamp())
        if t == "line_agg":
            a = g.groupby("_p")[y].mean().reset_index()
            return px.line(a, x="_p", y=y, markers=True, labels=dict(_p=x, **{y: f"mean {y}"})), f"Mean per {'month' if freq == 'M' else 'day'}."
        a = g.groupby("_p").size().reset_index(name="count")
        return px.bar(a, x="_p", y="count", labels=dict(_p=x)), f"Observations per {'month' if freq == 'M' else 'day'}."
    if t == "heatmap":
        ct = pd.crosstab(d[x].astype(str), d[y].astype(str))
        ct = ct.loc[ct.sum(1).nlargest(20).index, ct.sum(0).nlargest(20).index]
        if spec.get("norm"):
            return px.imshow(ct.div(ct.sum(1), axis=0) * 100, text_auto=".0f", aspect="auto", labels=dict(color="row %")), "Cells show % within each row."
        return px.imshow(ct, text_auto=True, aspect="auto"), ""
    return None, f"Unknown chart type {t}."
