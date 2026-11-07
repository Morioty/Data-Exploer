"""Synthetic, structurally different example datasets. `python -m ude.samples` writes them to ./examples."""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd


def environmental(n=400):
    r = np.random.default_rng(1)
    rain = r.gamma(1.2, 6, n)
    d = pd.DataFrame(dict(
        sensor_id=[f"S{i:04d}" for i in range(n)], date=pd.date_range("2022-01-01", periods=n, freq="D"),
        station=r.choice(["North", "South", "East", "West", "Ridge"], n), rainfall_mm=rain.round(2),
        soil_moisture=(25 + 0.6 * rain + r.normal(0, 4, n)).round(2),
        temperature_c=(15 + 10 * np.sin(np.arange(n) / 365 * 2 * np.pi) + r.normal(0, 2, n)).round(1),
        wind_speed=r.gamma(2, 3, n).round(1)))
    d.loc[r.choice(n, 30, replace=False), "soil_moisture"] = np.nan
    return d


def healthcare(n=600):
    r = np.random.default_rng(2)
    age = r.integers(18, 90, n)
    smoker = r.choice(["Yes", "No"], n, p=[0.3, 0.7])
    diag = [r.choice(["Hypertension", "Normal", "Other"], p=[.6, .2, .2] if s == "Yes" else [.2, .6, .2]) for s in smoker]
    return pd.DataFrame(dict(
        patient_id=np.arange(1000, 1000 + n), region=r.choice(["A", "B", "C", "D"], n), age=age,
        bmi=r.normal(26, 4, n).round(1), systolic_bp=(90 + 0.5 * age + r.normal(0, 10, n)).round(0),
        smoker=smoker, diagnosis=diag,
        report_month=pd.to_datetime("2023-01-01") + pd.to_timedelta(r.integers(0, 365, n), unit="D")))


def business(n=500):
    r = np.random.default_rng(3)
    seg = r.choice(["Consumer", "Corporate", "SMB"], n, p=[.5, .2, .3])
    mu = pd.Series(seg).map({"Consumer": 3.5, "Corporate": 5.0, "SMB": 4.2}).to_numpy()
    return pd.DataFrame(dict(
        customer_id=[f"C{i:05d}" for i in range(n)], segment=seg, channel=r.choice(["web", "store", "phone"], n),
        order_value=np.exp(r.normal(mu, 0.6)).round(2), quantity=r.poisson(3, n), discount=r.uniform(0, .3, n).round(3),
        region=r.choice(["EU", "US", "APAC"], n), order_date=pd.date_range("2022-01-01", periods=n, freq="D")))


def messy(n=300):
    r = np.random.default_rng(4)
    amount = r.normal(100, 20, n).round(2).astype(str).astype(object)
    amount[r.choice(n, 6, replace=False)] = "unknown"
    score = r.integers(0, 100, n).astype(str).astype(object)
    score[r.choice(n, 90, replace=False)] = "?"
    d = pd.DataFrame(dict(
        zip=[f"{v:05d}" for v in r.integers(0, 99999, n)], amount=amount, score=score,
        color=r.choice(["Red", "red ", "RED", "Blue", "blue", "Green"], n),
        when=r.choice(["2023-01-05", "05/02/2023", "2023-03-15 10:30", "2023/04/20"], n),
        country="US", empty=np.nan, note=[f"free text note number {i} about thing {i % 7}" for i in range(n)]))
    d.loc[:4, "zip"] = ["01234", "00501", "02108", "07001", "00100"]
    return pd.concat([d, d.iloc[:10]], ignore_index=True)


EXAMPLES = dict(environmental=environmental, healthcare=healthcare, business=business, messy=messy)

if __name__ == "__main__":
    out = pathlib.Path(__file__).resolve().parent.parent / "examples"
    out.mkdir(exist_ok=True)
    for k, f in EXAMPLES.items():
        f().to_csv(out / f"{k}.csv", index=False)
        print("wrote", out / f"{k}.csv")
