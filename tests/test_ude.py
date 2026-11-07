import json

import numpy as np
import pandas as pd
import pytest

from ude import cleaning as C
from ude import insights as I
from ude import quality as Q
from ude import report as R
from ude import stats as S
from ude.loading import LoadError, apply_types, infer_types, load_file
from ude.samples import EXAMPLES


def load(name):
    raw = load_file(f"{name}.csv", EXAMPLES[name]().to_csv(index=False).encode())
    kinds = infer_types(raw)
    return raw, kinds, apply_types(raw, kinds)


# ---------------------------------------------------------------- loading / types
def test_leading_zeros_preserved_and_identifier():
    raw, kinds, df = load("messy")
    assert kinds["zip"] == "identifier"
    assert df["zip"].iloc[0] == "01234"


def test_messy_types():
    raw, kinds, df = load("messy")
    assert kinds["when"] == "datetime" and kinds["amount"] == "numeric"
    assert kinds["color"] == "categorical" and kinds["note"] == "text"
    assert "amount" in Q.coercion_losses(raw, df)          # 'unknown' became missing, and is reported
    assert "score" in Q.mixed_type_columns(raw)


def test_other_samples_types():
    _, k, _ = load("environmental")
    assert k["sensor_id"] == "identifier" and k["date"] == "datetime" and k["rainfall_mm"] == "numeric"
    _, k, _ = load("healthcare")
    assert k["patient_id"] == "identifier" and k["region"] == "categorical"
    _, k, _ = load("business")
    assert k["segment"] == "categorical" and k["order_value"] == "numeric"


@pytest.mark.parametrize("name,data", [("a.exe", b"x"), ("a.csv", b""), ("a.csv", b"a,b\n"), ("a.json", b"{bad"), ("a.csv", b"\xff\xfe\x00bad,\xc3\x28\n1,2")])
def test_bad_files_give_load_error(name, data):
    with pytest.raises(LoadError):
        load_file(name, data)


def test_json_and_semicolon_csv():
    assert load_file("a.json", b'{"rows":[{"a":1,"b":"x"},{"a":2,"b":"y"}]}').shape == (2, 2)
    assert load_file("a.csv", b"a;b\n1;2\n3;4\n").shape == (2, 2)


def test_parquet_roundtrip(tmp_path):
    pytest.importorskip("pyarrow")
    p = tmp_path / "t.parquet"
    EXAMPLES["business"]().to_parquet(p)
    assert load_file("t.parquet", p.read_bytes()).shape[0] == 500


# ---------------------------------------------------------------- quality
def test_quality_metrics():
    raw, kinds, df = load("messy")
    assert Q.overview(df, kinds)["duplicates"] == 10
    assert "country" in Q.constant_columns(df)
    assert df["empty"].isna().all()
    assert "color" in Q.inconsistent_categories(df, kinds)
    texts = " ".join(w["text"] for w in Q.quality_warnings(df, kinds, raw, df))
    assert "duplicate" in texts and "entirely missing" in texts


def test_outliers_guards():
    assert Q.outliers(pd.Series([1, 1, 1, 1, 1, 1, 1, 1])) is None
    assert Q.outliers(pd.Series([1, 2])) is None
    o = Q.outliers(pd.Series(list(range(30)) + [1000]))
    assert o["iqr_count"] >= 1 and o["z_count"] >= 1


# ---------------------------------------------------------------- stats
def test_corr_and_guards():
    x = pd.Series(np.arange(50.0))
    d = pd.DataFrame({"x": x, "y": 2 * x + 1, "c": 1.0})
    r = S.corr_pair(d, "x", "y")
    assert r["ok"] and r["r"] == pytest.approx(1) and r["n"] == 50
    assert not S.corr_pair(d, "x", "c")["ok"]
    assert not S.corr_pair(d.head(5), "x", "y")["ok"]


def test_bh_monotone_and_bounded():
    p = np.array([0.001, 0.02, 0.03, 0.5])
    a = S.bh_adjust(p)
    assert (a >= p).all() and (a <= 1).all() and a[0] <= a[1] <= a[2] <= a[3]


def test_group_and_assoc_guards():
    d = pd.DataFrame({"g": ["a"] * 20 + ["b"] * 20, "v": list(range(20)) + list(range(100, 120))})
    r = S.group_compare(d, "g", "v")
    assert r["ok"] and r["eps2"] > 0.7
    assert not S.group_compare(d.assign(g="a"), "g", "v")["ok"]
    tiny = pd.DataFrame({"a": list("xyxy" * 6), "b": list("pqpq" * 6)})
    assert not S.cat_assoc(tiny.head(8), "a", "b")["ok"]


# ---------------------------------------------------------------- insights
def pairs(res):
    return {frozenset(i["vars"]) for i in res["insights"] if i["tested"]}


def test_insights_find_planted_relationships():
    for name, expect in [("environmental", {"rainfall_mm", "soil_moisture"}), ("healthcare", {"age", "systolic_bp"}),
                         ("business", {"segment", "order_value"}), ("healthcare", {"smoker", "diagnosis"})]:
        _, kinds, df = load(name)
        assert frozenset(expect) in pairs(I.discover(df, kinds)), name


def test_question_structure():
    _, kinds, df = load("environmental")
    res = I.discover(df, kinds)
    assert 3 <= len(res["questions"]) <= 7
    cols = set(df.columns)
    for q in res["questions"]:
        assert set(q["variables"]) <= cols and q["limitations"] and q["evidence"] and q["followup"]
        if q["followup_viz"]:
            assert {v for k, v in q["followup_viz"].items() if k in ("x", "y", "color") and v} <= cols
    for i in res["insights"]:
        assert i["n"] <= i["n_total"] and i["limitations"] and i["interpretation"]


def test_no_false_discoveries_on_noise():
    r = np.random.default_rng(0)
    df = pd.DataFrame(r.normal(size=(200, 8)), columns=list("abcdefgh"))
    res = I.discover(df, infer_types(df))
    assert not [i for i in res["insights"] if i["tested"]]


@pytest.mark.parametrize("df", [
    pd.DataFrame({"a": [1, 2, 3]}), pd.DataFrame({"a": [None] * 20}), pd.DataFrame({"a": ["x"] * 30}),
    pd.DataFrame({"a": list("abcdefghij" * 3), "b": list("pqrstuvwxy" * 3)}),
    pd.DataFrame({"n": np.arange(40.0), "m": np.arange(40.0)[::-1] * 2}),
])
def test_edge_datasets_do_not_crash(df):
    res = I.discover(df, infer_types(df))
    assert isinstance(res["questions"], list)


# ---------------------------------------------------------------- cleaning
def test_cleaning_replay_undo_and_original_untouched():
    _, kinds, df = load("messy")
    orig = df.copy()
    ops = C.add_op(df, [], dict(type="drop_duplicates"), "dedupe")
    ops = C.add_op(df, ops, dict(type="filter_rows", params=dict(column="color", op="contains", value="red")), "filter")
    cur = C.replay(df, ops)
    assert len(cur) < len(df) and cur["color"].str.lower().str.contains("red").all()
    assert len(C.replay(df, ops[:1])) == len(df) - 10
    pd.testing.assert_frame_equal(df, orig)
    assert json.loads(C.log_json("x.csv", df, ops))["operations"][0]["rows_before"] == len(df)


def test_cleaning_ops_and_errors():
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [" X ", "x", None]})
    assert C.apply_op(df, dict(type="fill_missing", params=dict(columns=["a"], strategy="median")))["a"].tolist() == [1, 2, 3]
    s = C.apply_op(df, dict(type="standardize_text", params=dict(columns=["b"], lower=True)))["b"]
    assert s.dropna().tolist() == ["x", "x"]
    with pytest.raises(ValueError):
        C.add_op(df, [], dict(type="rename", params=dict(mapping={"a": "b"})), "bad")
    with pytest.raises(ValueError):
        C.add_op(df, [], dict(type="fill_missing", params=dict(columns=["b"], strategy="mean")), "bad")


# ---------------------------------------------------------------- export
def test_report_escapes_untrusted_text():
    df = pd.DataFrame({"<script>alert(1)</script>": np.arange(40.0), "y": np.arange(40.0) * 2})
    kinds = infer_types(df)
    res = I.discover(df, kinds)
    html = R.html_report("<b>evil.csv</b>", df, Q.overview(df, kinds), Q.quality_warnings(df, kinds), res)
    assert "<script>alert(1)" not in html and "&lt;script&gt;" in html and "Limitations" in html


def test_summary_and_csv_export_not_sampled():
    _, kinds, df = load("business")
    assert len(S.summary_table(df, kinds)) == df.shape[1]
    assert pd.read_csv(__import__("io").BytesIO(R.csv_bytes(df))).shape[0] == len(df)
