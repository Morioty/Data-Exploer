"""Universal Data Explorer — run with: streamlit run app.py"""
import json

import pandas as pd
import streamlit as st

from ude import cleaning as C
from ude import insights as I
from ude import quality as Q
from ude import report as R
from ude import stats as S
from ude.loading import KINDS, LoadError, apply_types, excel_sheets, infer_types, load_file
from ude.samples import EXAMPLES
from ude.viz import make_fig

st.set_page_config(page_title="Universal Data Explorer", layout="wide")
ss = st.session_state


# ------------------------------------------------------------------ cached helpers
@st.cache_data(show_spinner=False, max_entries=4)
def c_infer(_raw, key):
    return infer_types(_raw)


@st.cache_data(show_spinner=False, max_entries=4)
def c_base(_raw, key, over):
    return apply_types(_raw, {**c_infer(_raw, key), **dict(over)})


@st.cache_data(show_spinner=False, max_entries=4)
def c_current(_base, key, over, ops_json):
    return C.replay(_base, json.loads(ops_json))


@st.cache_data(show_spinner="Searching for patterns…", max_entries=4)
def c_discover(_df, _kinds, sig):
    return I.discover(_df, _kinds)


_chart_n = [0]  # script re-runs top to bottom, so this resets every run


def plot(fig, note=""):
    if fig is None:
        st.info(note or "Chart not available for this selection.")
        return
    _chart_n[0] += 1
    st.plotly_chart(fig, width="stretch", key=f"chart{_chart_n[0]}")
    if note:
        st.caption(note)


# ------------------------------------------------------------------ sidebar: data source
with st.sidebar:
    st.title("Universal Data Explorer")
    up = st.file_uploader("Upload data", type=["csv", "tsv", "txt", "xlsx", "xls", "json", "parquet"],
                          help="Files are processed locally and never sent anywhere.")
    ex = st.selectbox("…or try an example", ["(none)"] + list(EXAMPLES))
    enc = st.selectbox("CSV/JSON encoding", ["utf-8", "utf-8-sig", "latin-1", "cp1252", "utf-16"])

if up is not None:
    name, data = up.name, up.getvalue()
elif ex != "(none)":
    name, data = f"{ex}.csv", EXAMPLES[ex]().to_csv(index=False).encode()
else:
    st.header("Upload a dataset to begin")
    st.write("Supports CSV, Excel, JSON and Parquet. You will get a structure overview, data-quality checks, "
             "interactive charts, and automatically generated, evidence-backed research questions.")
    st.stop()

sheet = None
if name.lower().endswith((".xlsx", ".xls")):
    try:
        sheets = excel_sheets(data)
    except LoadError as e:
        st.error(str(e))
        st.stop()
    with st.sidebar:
        sheet = st.selectbox("Excel sheet", sheets)

key = f"{name}|{len(data)}|{enc}|{sheet}"
if ss.get("key") != key:
    try:
        ss.raw = load_file(name, data, enc, sheet)
    except LoadError as e:
        st.error(str(e))
        st.stop()
    ss.update(key=key, over={}, ops=[], figs=[], name=name)

raw = ss.raw
over = tuple(sorted(ss.over.items()))
auto = c_infer(raw, key)
base = c_base(raw, key, over)
kinds0 = {**auto, **ss.over}
df = c_current(base, key, over, json.dumps(ss.ops, default=str))
# keep user/auto kinds for untouched columns; infer only for columns changed by cleaning
changed = [c for c in df.columns if c not in base or df[c].dtype != base[c].dtype]
kinds = {c: kinds0[c] for c in df.columns if c not in changed}
kinds.update(infer_types(df[changed]) if changed else {})
sig = f"{key}|{over}|{json.dumps(ss.ops, default=str)}"

if df.empty:
    st.warning("The current table has no rows. Reset cleaning steps in the Export tab.")
    st.stop()

t_over, t_explore, t_ins, t_export = st.tabs(["Overview", "Explore", "Insights", "Export"])
num_cols = [c for c in df if kinds[c] == "numeric"]
cat_cols = [c for c in df if kinds[c] in ("categorical", "boolean")]
dt_cols = [c for c in df if kinds[c] == "datetime"]
text_cols = [c for c in df if kinds[c] == "text"]
ov = Q.overview(df, kinds)
warns = Q.quality_warnings(df, kinds, raw if not ss.ops else None, base if not ss.ops else None)

# ================================================================== OVERVIEW
with t_over:
    a, b, c_, d_, e_ = st.columns(5)
    a.metric("Rows", f"{ov['rows']:,}")
    b.metric("Columns", ov["cols"])
    c_.metric("Memory", f"{ov['memory_mb']} MB")
    d_.metric("Missing cells", f"{ov['missing_pct']}%")
    e_.metric("Duplicate rows", ov["duplicates"], help="Rows identical in every column. Not removed unless you do it in Export.")
    st.caption("Variable types: " + ", ".join(f"{v} {k}" for k, v in ov["kind_counts"].items()))
    if ss.ops:
        st.info(f"Showing the cleaned table ({len(ss.ops)} step(s) applied). Original is preserved.")

    st.subheader("Data quality")
    if not warns:
        st.success("No data-quality issues detected.")
    for w in warns:
        (st.warning if w["level"] == "warning" else st.info)(w["text"])

    with st.expander("Correct variable types"):
        ed = st.data_editor(pd.DataFrame({"column": list(kinds0), "type": list(kinds0.values())}), hide_index=True, disabled=["column"],
                            column_config={"type": st.column_config.SelectboxColumn(options=KINDS)}, key="typed")
        new = dict(zip(ed["column"], ed["type"]))
        if new != kinds0 and st.button("Apply type changes (resets cleaning steps)"):
            ss.over = {c: k for c, k in new.items() if k != auto[c]}
            ss.ops = []
            st.rerun()
        st.caption("Identifiers and text are never used in statistics. Numeric conversion turns unparseable values into missing.")

    st.subheader("Preview")
    st.dataframe(df.head(100), width="stretch")
    st.subheader("Data dictionary")
    q = st.text_input("Search variables", placeholder="name, type or flag…")
    dd = Q.data_dictionary(df, kinds, Q.mixed_type_columns(raw) if not ss.ops else None)
    if q:
        dd = dd[dd.apply(lambda r: q.lower() in " ".join(map(str, r)).lower(), axis=1)]
    st.dataframe(dd, width="stretch", hide_index=True)

    st.subheader("Descriptive statistics")
    if num_cols:
        st.markdown("**Numeric**")
        st.dataframe(S.describe_numeric(df, num_cols).round(4), width="stretch", hide_index=True)
    if cat_cols:
        cc = st.selectbox("Categorical variable", cat_cols, key="dc")
        info, tb = S.describe_categorical(df, cc)
        st.write(f"{info['n']:,} values, {info['unique']} unique")
        st.dataframe(tb, hide_index=True)
    if dt_cols:
        dc = st.selectbox("Date variable", dt_cols, key="dd")
        info, tb = S.describe_datetime(df, dc)
        st.write(info)
        st.dataframe(tb, hide_index=True)
    if text_cols:
        tc = st.selectbox("Text variable", text_cols, key="dtx")
        info, tb = S.describe_text(df, tc)
        st.write(info)
        st.dataframe(tb, hide_index=True)
    if not (num_cols or cat_cols or dt_cols or text_cols):
        st.info("No variables suitable for descriptive statistics (identifiers are skipped).")

    with st.expander("Missing values and unusual numeric values"):
        mb = Q.missing_by_column(df)
        st.dataframe(mb[mb.missing > 0], hide_index=True)
        st.dataframe(Q.missing_patterns(df), hide_index=True)
        plot(*make_fig(df, dict(type="missing")))
        rows = [dict(variable=c, **o) for c in num_cols if (o := Q.outliers(df[c]))]
        if rows:
            st.dataframe(pd.DataFrame(rows).round(3), hide_index=True)
        st.caption(Q.OUTLIER_NOTE)

    with st.expander("Full profile report (fg-data-profiling, optional, slower)"):
        try:
            from data_profiling import ProfileReport
            import streamlit.components.v1 as components
            if st.button("Generate"):
                with st.spinner("Profiling…"):
                    components.html(ProfileReport(df, minimal=True, progress_bar=False).to_html(), height=800, scrolling=True)
        except ImportError:
            st.write("Install the optional package to enable this: `pip install -r requirements-profiling.txt`")

# ================================================================== EXPLORE
with t_explore:
    st.subheader("Chart builder")
    kind = st.selectbox("Chart", ["Histogram", "Box plot", "Bar chart", "Scatter", "Time series", "Correlation heatmap", "Contingency heatmap", "Missing values"])
    spec = None
    anyc = [c for c in df if kinds[c] not in ("identifier",)]
    if kind == "Histogram" and num_cols:
        spec = dict(type="hist", x=st.selectbox("Column", num_cols), color=st.selectbox("Colour by", [None] + [c for c in cat_cols if df[c].nunique() <= 10]))
    elif kind == "Box plot" and num_cols:
        spec = dict(type="box", y=st.selectbox("Numeric column", num_cols), x=st.selectbox("Group by", [None] + [c for c in cat_cols if df[c].nunique() <= 60]))
    elif kind == "Bar chart" and cat_cols:
        spec = dict(type="bar", x=st.selectbox("Column", cat_cols))
    elif kind == "Scatter" and len(num_cols) >= 2:
        x_ = st.selectbox("X", num_cols)
        y_ = st.selectbox("Y", [c for c in num_cols if c != x_])
        spec = dict(type="scatter", x=x_, y=y_, color=st.selectbox("Colour by", [None] + [c for c in cat_cols if df[c].nunique() <= 10]))
    elif kind == "Time series" and dt_cols and num_cols:
        spec = dict(type="line", x=st.selectbox("Date", dt_cols), y=st.selectbox("Value", num_cols))
    elif kind == "Correlation heatmap" and len(num_cols) >= 2:
        spec = dict(type="corr", cols=st.multiselect("Columns", num_cols, default=num_cols[:15]), method=st.radio("Method", ["spearman", "pearson"], horizontal=True))
    elif kind == "Contingency heatmap" and len(cat_cols) >= 2:
        a_ = st.selectbox("Rows", cat_cols)
        spec = dict(type="heatmap", x=a_, y=st.selectbox("Columns", [c for c in cat_cols if c != a_]), norm=st.checkbox("Row percentages", True))
    elif kind == "Missing values":
        spec = dict(type="missing")
    if spec is None:
        st.info("This chart needs columns of a type that this dataset does not have.")
    else:
        fig, note = make_fig(df, spec)
        plot(fig, note)
        if fig is not None and st.button("Save chart for export"):
            ss.figs.append((f"{kind}: {', '.join(str(v) for k, v in spec.items() if k in ('x', 'y', 'color') and v)}", fig))
            st.toast("Saved. Download it from the Export tab.")

    st.subheader("Relationship analysis")
    mode = st.radio("Compare", ["Number vs number", "Category vs number", "Category vs category"], horizontal=True)
    st.caption("Single test, no multiple-testing correction. Correlation and association do not imply causation.")
    if mode == "Number vs number" and len(num_cols) >= 2:
        x_ = st.selectbox("Variable 1", num_cols, key="r1")
        y_ = st.selectbox("Variable 2", [c for c in num_cols if c != x_], key="r2")
        rs = [S.corr_pair(df, x_, y_, m) for m in ("spearman", "pearson")]
        for r in rs:
            if r["ok"]:
                st.write(f"**{r['method'].title()}** r = {r['r']:.3f}, p = {r['p']:.3g}, valid pairs = {r['n']:,}")
            else:
                st.warning(r["reason"])
        plot(*make_fig(df, dict(type="scatter", x=x_, y=y_)))
    elif mode == "Category vs number" and cat_cols and num_cols:
        c_ = st.selectbox("Category", cat_cols, key="g1")
        n_ = st.selectbox("Number", num_cols, key="g2")
        r = S.group_compare(df, c_, n_)
        if r["ok"]:
            st.write(f"**{r['method']}** H = {r['H']:.2f}, p = {r['p']:.3g}, epsilon² = {r['eps2']:.3f} "
                     f"({S.effect_label('group', r['eps2'])}), n = {r['n']:,} in {r['groups']} groups. "
                     "Assumes independent observations; it does not show which groups differ.")
            st.dataframe(r["table"].round(3), hide_index=True)
        else:
            st.warning(r["reason"])
        plot(*make_fig(df, dict(type="box", x=c_, y=n_)))
    elif mode == "Category vs category" and len(cat_cols) >= 2:
        a_ = st.selectbox("Category 1", cat_cols, key="c1")
        b_ = st.selectbox("Category 2", [c for c in cat_cols if c != a_], key="c2")
        r = S.cat_assoc(df, a_, b_)
        if r["ok"]:
            st.write(f"**{r['method']}** chi² = {r['chi2']:.2f}, df = {r['dof']}, p = {r['p']:.3g}, "
                     f"Cramér's V = {r['cramers_v']:.3f} ({S.effect_label('assoc', r['cramers_v'])}), n = {r['n']:,}")
            st.dataframe(r["table"])
        else:
            st.warning(r["reason"])
    else:
        st.info("This comparison needs different variable types than this dataset has.")

    if st.toggle("Show automatic overview charts", value=False, help="Suggested charts based on variable types."):
        cols = st.columns(2)
        auto_specs = [dict(type="hist", x=c) for c in num_cols[:4]] + [dict(type="bar", x=c) for c in cat_cols[:3] if df[c].nunique() <= 60]
        if len(num_cols) >= 2:
            auto_specs.append(dict(type="corr", cols=num_cols))
        for i, sp in enumerate(auto_specs):
            with cols[i % 2]:
                st.caption(sp["type"] + (f": {sp['x']}" if "x" in sp else ""))
                plot(*make_fig(df, sp))

# ================================================================== INSIGHTS
with t_ins:
    res = c_discover(df, kinds, sig)
    st.caption(f"{res['n_tests']} statistical tests run; {res['n_passing']} findings passed correction and effect-size thresholds. "
               "Exploratory results: associations are not causes.")
    with st.expander("How findings are found and ranked"):
        for m in res["methods"]:
            st.markdown(f"- {m}")
    st.subheader("Suggested research questions")
    qs = res["questions"]
    if not qs:
        st.info("No analytical questions could be formed: the data has too few rows or usable variables for reliable tests.")
    else:
        pick = st.radio("Select a question to inspect its evidence", range(len(qs)), format_func=lambda i: qs[i]["question"])
        qd = qs[pick]
        by_id = {i["id"]: i for i in res["insights"]}
        st.markdown(f"**Variables:** " + " ".join(f"`{v}`" for v in qd["variables"]))
        st.markdown(f"**Why investigate:** {qd['reason']}")
        st.markdown(f"**Suggested analysis:** {qd['analysis']}")
        st.markdown(f"**Preliminary evidence:** {qd['evidence']}")
        spec_ = by_id[qd["insight_id"]]["viz"] if qd["insight_id"] in by_id else None
        if spec_ is None:
            kind_map = {"corr": "scatter", "group": "box", "assoc": "heatmap"}
            spec_ = dict(type=kind_map.get(qd["kind"], "bar"), x=qd["variables"][0], y=qd["variables"][1] if len(qd["variables"]) > 1 else None)
            if qd["kind"] == "assoc":
                spec_["norm"] = True
        plot(*make_fig(df, spec_))
        if qd["insight_id"] in by_id:
            ev = by_id[qd["insight_id"]]["evidence"]
            st.dataframe(pd.DataFrame(ev if isinstance(ev, list) else [ev]), hide_index=True)
            st.markdown(f"**Interpretation:** {by_id[qd['insight_id']]['interpretation']}")
        st.markdown("**Possible confounders (hypotheses, not established):**")
        for t in qd["confounders"]:
            st.markdown(f"- {t}")
        st.markdown("**Limitations:**")
        for t in qd["limitations"]:
            st.markdown(f"- {t}")
        st.markdown(f"**Suggested follow-up:** {qd['followup']}")
        if qd["followup_viz"]:
            plot(*make_fig(df, qd["followup_viz"]))

    st.subheader("All ranked findings")
    for n_, i in enumerate(res["insights"], 1):
        with st.expander(f"{n_}. {i['title']}"):
            st.write(f"**Variables:** {', '.join(i['vars'])}  \n**Method:** {i['method']}  \n**Statistic:** {i['stat']}  \n"
                     f"**Valid rows:** {i['n']:,} of {i['n_total']:,}  \n**Ranking score:** {i['score']} ({i['effect_name']} = {i['effect']:.2f})")
            st.write(i["interpretation"])
            plot(*make_fig(df, i["viz"]))
            st.caption("Limitations: " + " ".join(i["limitations"]))

# ================================================================== EXPORT
with t_export:
    st.subheader("Clean data")
    st.caption("Changes are applied to a copy; the original upload is kept. Nothing is applied until you confirm.")
    cur = df
    op = st.selectbox("Operation", C.OPS, format_func=lambda s: s.replace("_", " ").capitalize())
    params, desc = {}, op
    cols_all = list(cur.columns)
    if op == "drop_duplicates":
        desc = "Remove exact duplicate rows"
    elif op == "fill_missing":
        cs = st.multiselect("Columns", [c for c in cols_all if cur[c].isna().any()])
        how = st.selectbox("Strategy", ["drop_rows", "mean", "median", "mode", "constant"])
        val = st.text_input("Constant value") if how == "constant" else None
        params = dict(columns=cs, strategy=how, value=val)
        desc = f"Missing values ({how}) in {', '.join(cs)}"
    elif op == "rename":
        c0 = st.selectbox("Column", cols_all)
        nn = st.text_input("New name", value=c0)
        params = dict(mapping={c0: nn})
        desc = f"Rename {c0} to {nn}"
    elif op == "convert_type":
        c0 = st.selectbox("Column", cols_all)
        to = st.selectbox("Convert to", KINDS)
        params = dict(column=c0, to=to)
        desc = f"Convert {c0} to {to}"
    elif op == "filter_rows":
        c0 = st.selectbox("Column", cols_all)
        fo = st.selectbox("Condition", C.FILTER_OPS)
        fv = st.text_input("Value") if fo not in ("is missing", "is not missing") else None
        params = dict(column=c0, op=fo, value=fv)
        desc = f"Keep rows where {c0} {fo} {fv or ''}".strip()
    elif op == "select_columns":
        keep = st.multiselect("Columns to keep", cols_all, default=cols_all)
        params = dict(columns=keep)
        desc = f"Keep {len(keep)} columns"
    elif op == "standardize_text":
        cs = st.multiselect("Text columns", [c for c in cols_all if kinds[c] in ("categorical", "text", "identifier")])
        params = dict(columns=cs, strip=st.checkbox("Trim spaces", True), collapse=st.checkbox("Collapse repeated spaces", True), lower=st.checkbox("Lower-case", False))
        desc = f"Standardize text in {', '.join(cs)}"
    candidate = dict(type=op, params=params)
    try:
        prev = C.apply_op(cur, candidate)
        st.write(f"Preview: {len(cur):,} → {len(prev):,} rows, {cur.shape[1]} → {prev.shape[1]} columns")
        ok = True
    except Exception as e:
        st.warning(str(e))
        ok = False
    if st.checkbox("I confirm this change") and st.button("Apply", disabled=not ok):
        try:
            ss.ops = C.add_op(base, ss.ops, candidate, desc)
            st.rerun()
        except ValueError as e:
            st.error(str(e))
    if ss.ops:
        st.dataframe(pd.DataFrame([{k: o[k] for k in ("desc", "rows_before", "rows_after", "cols_before", "cols_after")} for o in ss.ops]), hide_index=True)
        u, r_ = st.columns(2)
        if u.button("Undo last step"):
            ss.ops = ss.ops[:-1]
            st.rerun()
        if r_.button("Reset to original"):
            ss.ops = []
            st.rerun()

    st.subheader("Download")
    st.caption(f"Exports contain all {len(df):,} rows; nothing is sampled.")
    stem = name.rsplit(".", 1)[0]
    c1, c2, c3 = st.columns(3)
    c1.download_button("Cleaned CSV", R.csv_bytes(df), f"{stem}_clean.csv", "text/csv")
    try:
        c1.download_button("Cleaned Excel", R.excel_bytes(df), f"{stem}_clean.xlsx")
    except RuntimeError as e:
        c1.caption(str(e))
    c2.download_button("Summary statistics (CSV)", R.csv_bytes(S.summary_table(df, kinds)), f"{stem}_summary.csv", "text/csv")
    c2.download_button("Transformation log (JSON)", C.log_json(name, base, ss.ops), f"{stem}_log.json", "application/json")
    res = c_discover(df, kinds, sig)
    c3.download_button("Insight report (HTML)", R.html_report(name, df, ov, warns, res, ss.ops, ss.figs), f"{stem}_report.html", "text/html")
    if ss.figs:
        c3.download_button(f"Saved charts ({len(ss.figs)}) (HTML)", R.figs_html(ss.figs), f"{stem}_charts.html", "text/html")
        if c3.button("Clear saved charts"):
            ss.figs = []
            st.rerun()
