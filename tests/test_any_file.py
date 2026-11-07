"""Hackathon-style inputs: odd encodings, containers, and less common formats."""
import gzip
import io
import sqlite3
import zipfile

import pandas as pd
import pytest

from ude import loading as L
from ude.samples import EXAMPLES

DF = EXAMPLES["business"]().head(60)
CSV = DF.to_csv(index=False).encode()


def same(df):
    assert df.shape == DF.shape and list(df.columns) == list(DF.columns)


def test_korean_cp949_and_utf16_csv():
    ko = pd.DataFrame({"지역": ["서울", "부산", "대구"] * 5, "값": range(15)})
    out = L.load_file("k.csv", ko.to_csv(index=False).encode("cp949"))          # default utf-8 fails -> falls back
    assert out["지역"].iloc[0] == "서울"
    out = L.load_file("k.csv", ko.to_csv(index=False).encode("utf-16"))
    assert out.shape == (15, 2)


def test_gzip_and_zip_containers():
    name, data = L.unwrap_gz("b.csv.gz", gzip.compress(CSV))
    assert name == "b.csv"
    same(L.load_file(name, data))
    same(L.load_file("b.csv.gz", gzip.compress(CSV)))                            # load_file also unwraps directly
    z = io.BytesIO()
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("readme.pdf", b"%PDF-1")
        zf.writestr("data/b.csv", CSV)
    assert L.zip_members(z.getvalue()) == ["data/b.csv"]
    same(L.load_file("b.csv", L.zip_read(z.getvalue(), "data/b.csv")))
    with pytest.raises(L.LoadError):
        L.load_file("x.zip", z.getvalue())


def test_decompression_bomb_is_capped(monkeypatch):
    monkeypatch.setattr(L, "MAX_BYTES", 1000)
    with pytest.raises(L.LoadError):
        L.unwrap_gz("b.csv.gz", gzip.compress(b"a,b\n" + b"1,2\n" * 5000))


def test_jsonl_xml_feather_stata_ods(tmp_path):
    jl = DF.to_json(orient="records", lines=True, date_format="iso").encode()
    same(L.load_file("b.jsonl", jl))
    same(L.load_file("b.json", jl))                                              # JSON Lines under a .json name
    xml = DF.to_xml(index=False).encode()
    same(L.load_file("b.xml", xml))
    b = io.BytesIO(); DF.to_feather(b)
    same(L.load_file("b.feather", b.getvalue()))
    s = DF.drop(columns=["order_date"]); b = io.BytesIO(); s.to_stata(b, write_index=False)
    assert L.load_file("b.dta", b.getvalue()).shape == s.shape
    pytest.importorskip("odf")
    b = io.BytesIO(); DF.to_excel(b, engine="odf", index=False)
    same(L.load_file("b.ods", b.getvalue()))


def test_sqlite_tables_and_picker(tmp_path):
    p = tmp_path / "t.db"
    con = sqlite3.connect(p)
    DF.to_sql("orders", con, index=False)
    pd.DataFrame({"a": [1, 2]}).to_sql("tiny", con, index=False)
    con.commit(); con.close()
    data = p.read_bytes()
    assert L.table_names("t.db", data) == ["orders", "tiny"]
    assert L.load_file("t.db", data, sheet="orders").shape == DF.shape
    assert L.load_file("t.sqlite", data, sheet="tiny").shape == (2, 1)
    with pytest.raises(L.LoadError):
        L.load_file("t.db", data, sheet='x"; DROP TABLE orders; --')            # name must exist in the database


@pytest.mark.parametrize("name", ["data.dat", "data", "export.tab"])
def test_unknown_extension_is_sniffed(name):
    same(L.load_file(name, CSV))
    assert L.load_file(name, b"a;b\n1;2\n3;4\n").shape == (2, 2)


def test_unusable_files_get_clear_errors():
    for name, data in [("r.pdf", b"%PDF-1.4 ..."), ("pic.png", b"\x89PNG"), ("x.bin", b"\x00\x01\x02\x03" * 100), ("t.dat", b"just one line of prose")]:
        with pytest.raises(L.LoadError):
            L.load_file(name, data)


def test_undecodable_bytes_fall_back_instead_of_failing():
    assert L.load_file("a.csv", b"a,b\n\xc3\x28,2\n3,4\n").shape == (2, 2)
