# Universal Data Explorer

Local, dataset-agnostic EDA plus a deterministic analysis assistant that turns verified statistics into ranked findings and research questions. No internet, API keys or LLMs. No predictive modelling.

## Install and run (Python 3.11+)

```
cd universal-data-explorer
python -m venv .venv && .venv\Scripts\activate      # Windows; use source .venv/bin/activate elsewhere
pip install -r requirements.txt
streamlit run app.py
```

Optional full profile report (adds heavy dependencies: numba, matplotlib, statsmodels…):
`pip install -r requirements-profiling.txt`

Tests: `pytest`. Example CSVs: `python -m ude.samples` (writes `examples/`); the app also offers them in the sidebar.

## Reuse vs. custom (reviewed 2026-11)

| Project | Decision | Reason |
|---|---|---|
| fg-data-profiling (MIT, active) | Reused, optional | Deep per-column profile on demand. Heavy dependencies, so not required. |
| PyGWalker | Not used | Telemetry packages; offline operation unverified. Plotly covers the chart builder. |
| Lux | Not used | Stale (last release 2022), Jupyter-only, patches pandas. Idea reused: generate, score, rank. |
| AutoViz | Not used | Stale; depends on xgboost/scikit-learn/nltk/holoviews. |
| RATH | Not used | AGPL-3.0, a TypeScript app rather than a library. |

Custom (`ude/`): `loading` (import, type inference, overrides), `quality`, `stats` (guarded tests), `insights` (the engine), `viz`, `cleaning` (replayable ops), `report` (exports).

## How insights are produced

All pair tests run (Spearman, Kruskal-Wallis, chi-square, Spearman-with-time), all p-values are Benjamini-Hochberg adjusted together, and a finding needs adjusted p < 0.05 **and** a minimum effect size. Ranking score = strength × completeness × sample factor: an attention-ordering, not a probability or causal importance. Tests that would be invalid (too few rows, constant column, sparse tables, identifiers) are skipped, never faked.

## Known limits

- Dates are parsed with pandas' mixed-format parser; day/month ambiguity (05/02/2023) is not resolved.
- Rows are treated as independent; repeated measures inflate significance.
- Cleaning replays from the original on each change, which is simple but slower on very large tables.
- Exports do not neutralise spreadsheet formulas (cells starting with `=`); treat exported files from untrusted sources with care in Excel.
- Static PNG chart export is not included (needs `kaleido`); charts export as interactive HTML.
