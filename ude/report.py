"""Exports: CSV/Excel, summary stats, HTML insight report. All untrusted text is HTML-escaped."""
from __future__ import annotations

import io
from html import escape as e

import pandas as pd

from .insights import METHODS
from .viz import make_fig

CSS = ("body{font-family:system-ui,sans-serif;max-width:900px;margin:2em auto;padding:0 1em;color:#222}"
       "table{border-collapse:collapse;font-size:.85em}td,th{border:1px solid #ccc;padding:3px 8px}"
       ".card{border:1px solid #ddd;border-radius:6px;padding:.6em 1em;margin:1em 0}.muted{color:#666}")


def csv_bytes(df) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def excel_bytes(df) -> bytes:
    b = io.BytesIO()
    try:
        df.to_excel(b, index=False)
    except ImportError:
        raise RuntimeError("Excel export needs the 'openpyxl' package (pip install openpyxl).")
    return b.getvalue()


def _table(rows) -> str:
    if isinstance(rows, dict):
        rows = [rows]
    if not rows:
        return ""
    return pd.DataFrame(rows).to_html(index=False, escape=True, border=0)


def html_report(source, df, overview, warnings, result, ops=(), figs=()) -> str:
    h = [f"<h1>Data exploration report</h1><p class=muted>Source: {e(source)} &middot; {overview['rows']:,} rows &times; {overview['cols']} columns"
         f" &middot; {overview['missing_pct']}% missing &middot; {overview['duplicates']} duplicate rows</p>"]
    h.append("<h2>Data quality</h2><ul>" + "".join(f"<li>{e(w['text'])}</li>" for w in warnings) + "</ul>" if warnings else "<h2>Data quality</h2><p>No warnings.</p>")
    h.append("<h2>Methods</h2><ul>" + "".join(f"<li>{e(m)}</li>" for m in METHODS) + "</ul>")
    h.append(f"<p>{result['n_tests']} statistical tests were run; {result['n_passing']} met the reporting threshold after correction.</p>")
    h.append("<h2>Research questions</h2>")
    for q in result["questions"]:
        h.append(f"<div class=card><b>{e(q['question'])}</b><br>Variables: {e(', '.join(q['variables']))}<br>Why: {e(q['reason'])}"
                 f"<br>Suggested analysis: {e(q['analysis'])}<br>Evidence: {e(q['evidence'])}"
                 f"<br>Possible confounders: {e(' '.join(q['confounders']))}<br>Follow-up: {e(q['followup'])}"
                 f"<br><span class=muted>Limitations: {e(' '.join(q['limitations']))}</span></div>")
    h.append("<h2>Ranked insights</h2>")
    for i in result["insights"]:
        p = f", adjusted p {i['p_adj']:.3g}" if i["p_adj"] is not None else ""
        h.append(f"<div class=card><b>{e(i['title'])}</b><br>Variables: {e(', '.join(i['vars']))} &middot; Method: {e(i['method'])}"
                 f"<br>Statistic: {e(i['stat'])}{e(p)} &middot; valid n = {i['n']} of {i['n_total']} &middot; ranking score {i['score']}"
                 f"<br>{e(i['interpretation'])}<br><span class=muted>Limitations: {e(' '.join(i['limitations']))}</span>"
                 f"{_table(i['evidence'])}</div>")
    if ops:
        h.append("<h2>Transformations applied</h2><ol>" + "".join(f"<li>{e(o['desc'])} ({o['rows_before']}&rarr;{o['rows_after']} rows)</li>" for o in ops) + "</ol>")
    if figs:
        h.append("<h2>Selected charts</h2>")
        for k, (title, fig) in enumerate(figs):
            h.append(f"<h3>{e(title)}</h3>" + fig.to_html(full_html=False, include_plotlyjs=True if k == 0 else False))
    h.append("<h2>Limitations</h2><p>Results are exploratory. Associations are not causal. Multiple-testing correction reduces but does not remove false discoveries. "
             "Unusual values were never removed.</p>")
    return f"<!doctype html><meta charset=utf-8><title>Data exploration report</title><style>{CSS}</style>" + "".join(h)


def figs_html(figs) -> str:
    parts = [f"<h2>{e(t)}</h2>" + f.to_html(full_html=False, include_plotlyjs=True if k == 0 else False) for k, (t, f) in enumerate(figs)]
    return f"<!doctype html><meta charset=utf-8><title>Charts</title>" + "".join(parts)
