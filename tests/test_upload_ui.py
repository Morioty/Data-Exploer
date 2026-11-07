import io, gzip, zipfile, sqlite3, tempfile, os
import pathlib
import pytest
from streamlit.testing.v1 import AppTest
from ude.samples import EXAMPLES

D = EXAMPLES["business"]()
NL = chr(10)


def mk():
    out = {}
    out["a.csv"] = D.to_csv(index=False).encode()
    out["a.csv.gz"] = gzip.compress(out["a.csv"])
    b = io.BytesIO(); D.to_excel(b, index=False); out["a.xlsx"] = b.getvalue()
    out["a.json"] = D.to_json(orient="records").encode()
    b = io.BytesIO(); D.to_parquet(b); out["a.parquet"] = b.getvalue()
    z = io.BytesIO()
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("d/a.csv", out["a.csv"])
    out["a.zip"] = z.getvalue()
    p = os.path.join(tempfile.mkdtemp(), "t.db")
    c = sqlite3.connect(p); D.to_sql("o", c, index=False); c.commit(); c.close()
    out["a.db"] = open(p, "rb").read()
    out["a.html"] = D.head(50).to_html(index=False).encode()
    h = list(D.columns)
    rows = ["|" + "|".join(map(str, r)) + "|" for r in D.head(50).itertuples(index=False)]
    out["a.md"] = (NL.join(["|" + "|".join(h) + "|", "|" + "|".join("---" for _ in h) + "|"] + rows)).encode()
    out["data.dat"] = out["a.csv"]
    return out


APP = str(pathlib.Path(__file__).resolve().parent.parent / "app.py")
FILES = mk()


@pytest.mark.parametrize("name", list(FILES))
def test_upload_through_the_real_app(name):
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.main.file_uploader[0].upload(name, FILES[name])
    at.run()
    assert not at.exception and not at.error
    assert at.metric[0].value in ("500", "50")
