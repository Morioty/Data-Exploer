"""Every example dataset x every supported file format goes through the same path the web app uses."""
import io

import pandas as pd
import pytest

from ude import insights as I
from ude import quality as Q
from ude import report as R
from ude import stats as S
from ude.loading import apply_types, infer_types, load_file
from ude.samples import EXAMPLES


def encode(df, fmt):
    if fmt == "csv":
        return df.to_csv(index=False).encode()
    b = io.BytesIO()
    if fmt == "xlsx":
        df.to_excel(b, index=False)
    elif fmt == "parquet":
        df.to_parquet(b)
    elif fmt == "json":
        return df.to_json(orient="records", date_format="iso").encode()
    return b.getvalue()


@pytest.mark.parametrize("fmt", ["csv", "xlsx", "json", "parquet"])
@pytest.mark.parametrize("name", list(EXAMPLES))
def test_full_pipeline(name, fmt):
    src = EXAMPLES[name]()
    raw = load_file(f"{name}.{fmt}", encode(src, fmt))
    assert raw.shape == src.shape
    kinds = infer_types(raw)
    df = apply_types(raw, kinds)
    ov = Q.overview(df, kinds)
    res = I.discover(df, kinds)
    assert res["questions"] or name == "messy"
    html = R.html_report(f"{name}.{fmt}", df, ov, Q.quality_warnings(df, kinds), res)
    assert "Limitations" in html
    assert len(S.summary_table(df, kinds)) == df.shape[1]
    assert pd.read_csv(io.BytesIO(R.csv_bytes(df))).shape[0] == len(df)
    R.excel_bytes(df)


# ---------------------------------------------------------------- HTML / Markdown tables
from ude.loading import LoadError, table_names  # noqa: E402

HTML = b"""<html><body><script>alert(1)</script>
<table><tr><th>nav</th></tr><tr><td>x</td></tr></table>
<table><thead><tr><th>zip</th><th>score</th></tr></thead>
<tr><td>01234</td><td>5</td></tr><tr><td>00501</td><td>7</td></tr><tr><td></td><td>9</td></tr></table></body></html>"""
MD = rb"""# Report

| zip | score |
|-----|:-----:|
| 01234 | 5 |
| 00501 | a \| b |
| 02108 | 9 |

text

| a | b |
|---|---|
| 1 | 2 |
"""


@pytest.mark.parametrize("name,data", [("t.html", HTML), ("t.md", MD)])
def test_html_md_tables(name, data):
    names = table_names(name, data)
    assert len(names) == 2 and "3 rows" in names[0]          # largest first
    df = load_file(name, data)                               # default = largest
    assert df.shape == (3, 2) and df["zip"].iloc[0] == "01234"   # leading zeros kept
    small = load_file(name, data, sheet=names[1])
    assert small.shape[1] in (1, 2) and len(small) in (1, 1)
    assert infer_types(df)["zip"] == "identifier" if name == "t.html" else True


def test_md_escaped_pipe_and_no_tables():
    assert "a | b" in load_file("t.md", MD)["score"].tolist()
    with pytest.raises(LoadError):
        load_file("t.html", b"<html><p>no tables</p></html>")
    with pytest.raises(LoadError):
        load_file("t.md", b"just some text")
