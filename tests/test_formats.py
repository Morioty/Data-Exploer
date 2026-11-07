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
