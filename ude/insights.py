"""Deterministic insight engine (architecture inspired by Lux/RATH: generate candidates -> test -> score -> rank).
Only reports what was computed. All p-values are Benjamini-Hochberg adjusted across ALL tests run."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import stats as S
from .quality import outliers

ALPHA = 0.05
THRESH = dict(corr=0.3, trend=0.3, assoc=0.2, group=0.08, skew=1.0, imbalance=0.8, outlier_pct=1.0, missing=0.2)
CAPS = dict(num=25, cat=12, dt=2)

METHODS = [
    "Candidates: every pair of numeric columns (Spearman rank correlation), every categorical-by-numeric pair "
    "(Kruskal-Wallis, epsilon-squared), every categorical pair (chi-square, Cramer's V), and each date-by-numeric pair "
    "(Spearman with time). Identifiers, free text and constant columns are never tested.",
    "Multiple testing: all p-values from all tests are adjusted together with Benjamini-Hochberg (false-discovery rate). "
    f"A finding is reported as tested only if adjusted p < {ALPHA} AND its effect size passes a minimum "
    f"(|rho|>={THRESH['corr']}, epsilon^2>={THRESH['group']}, V>={THRESH['assoc']}), so tiny effects in big data are not promoted.",
    "Ranking score = strength x completeness x sample factor. Strength is |rho|, sqrt(epsilon^2) or Cramer's V "
    "(all on a 0-1 scale); completeness = rows used / total rows; sample factor = min(1, rows used / 100). "
    "It orders findings for your attention. It is NOT a probability, a causal importance, or real-world impact.",
    "Tested findings are ranked above descriptive notes (skew, imbalance, unusual values, gaps), which have no p-value.",
    "Large datasets: tests use all rows; only charts are sampled (and say so).",
]


def _p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _pick(df, kinds, kindset, cap, levels=None, min_n=S.MIN_N):
    out = []
    for c in df.columns:
        if kinds.get(c) not in kindset:
            continue
        s = df[c].dropna()
        nu = s.nunique()
        if len(s) < min_n or nu < 2 or (levels and not levels[0] <= nu <= levels[1]):
            continue
        out.append((len(s), c))
    return [c for _, c in sorted(out, key=lambda t: -t[0])[:cap]]


# ------------------------------------------------------------------ tests
def run_tests(df, kinds) -> list[dict]:
    num = _pick(df, kinds, {"numeric"}, CAPS["num"])
    cat = _pick(df, kinds, {"categorical", "boolean"}, CAPS["cat"], levels=(2, S.MAX_LEVELS))
    dt = _pick(df, kinds, {"datetime"}, CAPS["dt"], min_n=S.MIN_N)
    tests = []

    def add(kind, vars_, r, effect, ename, raw=None):
        if r["ok"]:
            tests.append(dict(kind=kind, vars=vars_, p=r["p"], n=r["n"], effect=effect, effect_name=ename, res=r))

    for i, a in enumerate(num):
        for b in num[i + 1:]:
            r = S.corr_pair(df, a, b, "spearman")
            add("corr", (a, b), r, abs(r.get("r", 0)), "|Spearman rho|")
        for t in dt:
            r = S.trend(df, t, a)
            add("trend", (t, a), r, abs(r.get("r", 0)), "|Spearman rho with time|")
    for c in cat:
        for a in num:
            r = S.group_compare(df, c, a)
            add("group", (c, a), r, np.sqrt(r.get("eps2", 0)), "sqrt(epsilon^2)")
    for i, a in enumerate(cat):
        for b in cat[i + 1:]:
            r = S.cat_assoc(df, a, b)
            add("assoc", (a, b), r, r.get("cramers_v", 0), "Cramer's V")
    adj = S.bh_adjust([t["p"] for t in tests])
    for t, a in zip(tests, adj):
        t["p_adj"] = float(a)
    return tests


def _passes(t) -> bool:
    th = {"corr": THRESH["corr"], "trend": THRESH["trend"], "assoc": THRESH["assoc"], "group": np.sqrt(THRESH["group"])}[t["kind"]]
    return t["p_adj"] < ALPHA and t["effect"] >= th


def _score(effect, n, N):
    return round(float(effect) * (n / N if N else 0) * min(1.0, n / 100), 4)


def _loss(n, N):
    return f"{N - n} of {N} rows were unusable here because of missing values." if N > n else "All rows were usable."


# ------------------------------------------------------------ explanations
def from_test(t, N, df=None) -> dict:
    k, v, r, n = t["kind"], t["vars"], t["res"], t["n"]
    a, b = v
    pa = _p(t["p_adj"])
    weak = "" if _passes(t) else "Did not meet the reporting threshold (after multiple-testing correction / minimum effect). "
    base = dict(id=f"{k}:{a}|{b}", kind=k, vars=list(v), n=n, n_total=N, p=t["p"], p_adj=t["p_adj"],
                effect=t["effect"], effect_name=t["effect_name"], tested=True, passed=_passes(t),
                score=_score(t["effect"], n, N))
    if k == "corr":
        pe = S.corr_pair(df, a, b, "pearson") if df is not None else None
        lab = S.effect_label("corr", t["effect"])
        d = "higher" if r["r"] > 0 else "lower"
        base.update(
            title=f"{a} vs {b}: {lab} {'positive' if r['r'] > 0 else 'negative'} monotonic association",
            method="Spearman rank correlation", stat=f"rho = {r['r']:.2f}, adjusted p = {pa}, n = {n}",
            interpretation=f"{weak}Across {n} rows with both values, higher {a} tends to go with {d} {b} (Spearman rho = {r['r']:.2f}, "
                           f"adjusted p = {pa}). This describes co-movement only; it does not show that either causes the other.",
            limitations=["Rank correlation detects monotonic patterns only; check the scatter for curves or clusters.",
                         "Outliers, mixed subgroups or a shared third variable can create or hide an association.", _loss(n, N)],
            viz=dict(type="scatter", x=a, y=b), evidence={"Spearman rho": round(r["r"], 3),
                     "Pearson r": round(pe["r"], 3) if pe and pe["ok"] else "n/a", "valid pairs": n})
    elif k == "trend":
        lab = S.effect_label("trend", t["effect"])
        base.update(
            title=f"{b} over time ({a}): {lab} {'upward' if r['r'] > 0 else 'downward'} drift",
            method="Spearman correlation with time", stat=f"rho = {r['r']:.2f}, adjusted p = {pa}, n = {n}",
            interpretation=f"{weak}{b} tends to {'increase' if r['r'] > 0 else 'decrease'} with {a} (rho = {r['r']:.2f}, adjusted p = {pa}, n = {n}). "
                           "This is a monotonic tendency in the observed period, not a forecast.",
            limitations=["Assumes observations are comparable over time; irregular sampling or changing locations can mimic a trend.",
                         "Seasonality and autocorrelation are not modelled, so the p-value is optimistic.", _loss(n, N)],
            viz=dict(type="line", x=a, y=b), evidence={"Spearman rho": round(r["r"], 3), "valid rows": n})
    elif k == "group":
        tb = r["table"]
        hi, lo = tb.iloc[0], tb.iloc[-1]
        lab = S.effect_label("group", r["eps2"])
        base.update(
            title=f"{b} differs across {a} groups ({lab} effect)",
            method="Kruskal-Wallis test", stat=f"H = {r['H']:.1f}, epsilon^2 = {r['eps2']:.2f}, adjusted p = {pa}, n = {n}, groups = {r['groups']}",
            interpretation=f"{weak}{b} differs between {a} groups (H = {r['H']:.1f}, epsilon^2 = {r['eps2']:.2f}, adjusted p = {pa}). "
                           f"Median {b} is highest for '{hi[a]}' ({hi['median']:.3g}, n={int(hi['n'])}) and lowest for '{lo[a]}' ({lo['median']:.3g}, n={int(lo['n'])}). "
                           "The test shows groups differ somewhere, not which pairs differ or why.",
            limitations=["Observations are assumed independent (repeated measures violate this).",
                         "Groups with fewer than 3 observations were excluded" + (f" ({r['dropped_groups']} dropped)." if r["dropped_groups"] else "."),
                         "A difference between groups may reflect other variables unevenly spread across them.", _loss(n, N)],
            viz=dict(type="box", x=a, y=b), evidence=tb.round(3).to_dict("records"))
    else:
        lab = S.effect_label("assoc", r["cramers_v"])
        base.update(
            title=f"{a} and {b} are associated ({lab})",
            method="Chi-square test of independence", stat=f"chi2 = {r['chi2']:.1f}, df = {r['dof']}, Cramer's V = {r['cramers_v']:.2f}, adjusted p = {pa}, n = {n}",
            interpretation=f"{weak}The mix of {b} categories differs across {a} categories (Cramer's V = {r['cramers_v']:.2f}, adjusted p = {pa}, n = {n}). "
                           "Association is not causation and says nothing about direction.",
            limitations=["Cell counts passed the minimum-expected-count check, but sparse cells still make V noisy.",
                         "Cramer's V is biased upward in small samples.", _loss(n, N)],
            viz=dict(type="heatmap", x=a, y=b, norm=True), evidence=r["table"].reset_index().to_dict("records"))
    return base


def descriptive(df, kinds, N) -> list[dict]:
    out = []
    for c in df.columns:
        s, k = df[c].dropna(), kinds.get(c)
        n = len(s)
        miss = 1 - n / N if N else 0
        if miss >= THRESH["missing"] and n < N:
            out.append(dict(id=f"missing:{c}", kind="missing", vars=[c], n=n, n_total=N, tested=False, passed=True, effect=miss,
                            effect_name="missing share", p=None, p_adj=None, score=_score(miss, max(n, 1), N),
                            title=f"{c} is {100 * miss:.0f}% missing", method="Missing-value count",
                            stat=f"{N - n} of {N} rows missing", interpretation=f"Only {n} of {N} rows have a value for {c}, so any analysis using it covers a limited subset.",
                            limitations=["Whether values are missing at random is unknown; the missing rows may differ systematically."],
                            viz=dict(type="missing"), evidence={"missing rows": N - n}))
        if k == "numeric" and n >= 20 and s.nunique() >= 3:
            sk = float(pd.to_numeric(s).skew())
            if abs(sk) >= THRESH["skew"]:
                out.append(dict(id=f"skew:{c}", kind="skew", vars=[c], n=n, n_total=N, tested=False, passed=True, effect=min(abs(sk) / 3, 1),
                                effect_name="|skew|/3 (capped)", p=None, p_adj=None, score=_score(min(abs(sk) / 3, 1), n, N),
                                title=f"{c} is {'right' if sk > 0 else 'left'}-skewed", method="Sample skewness", stat=f"skewness = {sk:.2f}, n = {n}",
                                interpretation=f"{c} has a long {'right' if sk > 0 else 'left'} tail (skewness {sk:.2f}); mean ({s.mean():.3g}) and median ({s.median():.3g}) differ, so the mean may not represent a typical value.",
                                limitations=["Skewness is sensitive to a few extreme values.", "Consider rank-based methods or a log view (this tool does not transform data silently)."],
                                viz=dict(type="hist", x=c), evidence={"mean": round(float(s.mean()), 3), "median": round(float(s.median()), 3), "skew": round(sk, 3)}))
            o = outliers(s)
            if o and n >= 30 and o["iqr_pct"] >= THRESH["outlier_pct"] and not o["iqr_is_zero"]:
                e = min(o["iqr_pct"] / 10, 1)
                out.append(dict(id=f"outlier:{c}", kind="outlier", vars=[c], n=n, n_total=N, tested=False, passed=True, effect=e,
                                effect_name="outlier share/10% (capped)", p=None, p_adj=None, score=_score(e, n, N),
                                title=f"{c} has unusual values ({o['iqr_pct']}% outside IQR fences)", method="IQR rule (1.5x) and |z|>3",
                                stat=f"IQR: {o['iqr_count']} values, z-score: {o['z_count']} values, n = {n}",
                                interpretation=f"{o['iqr_count']} of {n} values lie outside [{o['iqr_lower']:.3g}, {o['iqr_upper']:.3g}]. They may be genuine extremes or recording issues; none were removed.",
                                limitations=["IQR over-flags skewed data; z-scores are distorted by the outliers themselves.", "Unusual does not mean wrong."],
                                viz=dict(type="box", y=c), evidence={k_: v for k_, v in o.items() if k_ in ("iqr_count", "z_count", "iqr_lower", "iqr_upper")}))
        if k in ("categorical", "boolean") and n >= 20 and s.nunique() >= 2:
            vc = s.astype(str).value_counts(normalize=True)
            if vc.iloc[0] >= THRESH["imbalance"]:
                out.append(dict(id=f"imbalance:{c}", kind="imbalance", vars=[c], n=n, n_total=N, tested=False, passed=True, effect=float(vc.iloc[0]),
                                effect_name="top category share", p=None, p_adj=None, score=_score(float(vc.iloc[0]), n, N),
                                title=f"{c} is dominated by '{vc.index[0]}' ({100 * vc.iloc[0]:.0f}%)", method="Category proportions",
                                stat=f"top share = {vc.iloc[0]:.2f}, categories = {len(vc)}, n = {n}",
                                interpretation=f"'{vc.index[0]}' makes up {100 * vc.iloc[0]:.0f}% of {c}; the other {len(vc) - 1} categories share the rest, so comparisons involving them rest on few rows.",
                                limitations=["Rare categories give unstable estimates.", "The sample may not reflect the real mix."],
                                viz=dict(type="bar", x=c), evidence={"top": vc.index[0], "share": round(float(vc.iloc[0]), 3)}))
        if k == "datetime" and n >= 10:
            g = s.sort_values().diff().dropna()
            med = g.median()
            if med > pd.Timedelta(0) and g.max() > 5 * med:
                st_ = s.sort_values().iloc[(g.values.argmax())]
                out.append(dict(id=f"gap:{c}", kind="gap", vars=[c], n=n, n_total=N, tested=False, passed=True, effect=0.3,
                                effect_name="fixed (descriptive)", p=None, p_adj=None, score=_score(0.3, n, N),
                                title=f"{c} has a coverage gap of {g.max()}", method="Largest gap vs median spacing",
                                stat=f"largest gap = {g.max()}, median gap = {med}, n = {n}",
                                interpretation=f"After {st_} there is a {g.max()} gap, more than 5x the typical spacing ({med}). Trends spanning it may be unreliable.",
                                limitations=["The gap may be genuine (no events) or a collection failure."],
                                viz=dict(type="line_agg_count", x=c), evidence={"largest_gap": str(g.max()), "median_gap": str(med)}))
    return out


# ------------------------------------------------------------ questions
def _confounders(a, b, passing):
    sig = {frozenset(t["vars"]) for t in passing}
    others = {c for t in passing for c in t["vars"]} - {a, b}
    return [c for c in sorted(others) if frozenset((a, c)) in sig and frozenset((b, c)) in sig][:3]


def _first_cat_for(df, kinds, cols):
    for c in df.columns:
        if kinds.get(c) in ("categorical", "boolean") and c not in cols and 2 <= df[c].nunique() <= 8:
            return c


def make_question(ins, passing_tests, df, kinds) -> dict:
    k, v = ins["kind"], ins["vars"]
    a, b = (v + [None, None])[:2]
    conf = _confounders(a, b, passing_tests) if b else []
    conf_txt = ([f"'{c}' is associated with both {a} and {b} here and could partly account for their relationship (not tested causally)." for c in conf]
                or ["No third variable in this dataset stands out, but unmeasured factors (e.g. location, time, subgroup) may still matter."])
    cc = _first_cat_for(df, kinds, v)
    q, an, fu, fviz = {
        "corr": (f"How does {b} vary with {a}, and does the pattern hold within subgroups?",
                 "Spearman and Pearson correlation, scatter plot; repeat within groups of a categorical variable.",
                 (f"Colour the scatter by '{cc}' to see whether the pattern holds within its groups." if cc else "Re-check the correlation with the unusual values of each variable set aside (view only)."),
                 dict(type="scatter", x=a, y=b, color=cc) if cc else None),
        "trend": (f"Does {b} change over time ({a}), and is the change steady?",
                  "Spearman correlation with time; monthly/daily mean plot; check coverage gaps.",
                  f"View the period-averaged {b} to separate steady drift from seasonality or gaps.",
                  dict(type="line_agg", x=a, y=b)),
        "group": (f"Which {a} groups differ most in {b}, and why might they differ?",
                  "Kruskal-Wallis test, group medians, box plot; pairwise follow-ups with correction for the largest gaps.",
                  f"Compare the distribution of {b} across {a} groups as overlaid histograms to see shape, not just centre.",
                  dict(type="hist", x=b, color=a)),
        "assoc": (f"Are {a} and {b} related, and in which category combinations?",
                  "Chi-square test and the contingency table; compare row percentages.",
                  f"View row percentages of {b} within each {a} category to see which combinations drive the association.",
                  dict(type="heatmap", x=a, y=b, norm=True)),
        "skew": (f"Do a few extreme values drive the summary of {a}?", "Histogram, mean vs median, IQR outlier review.",
                 f"Re-draw {a} on a log scale (view only) to see the bulk of the data.", dict(type="hist", x=a, log_x=True)),
        "outlier": (f"Are the unusual values of {a} genuine observations or recording issues?", "Box plot, IQR and z-score listing, check against domain limits.",
                    f"Inspect the distribution of {a} for isolated points versus a long tail.", dict(type="hist", x=a)),
        "imbalance": (f"What explains the dominance of one category in {a}, and does it limit comparisons?", "Frequency table and group-size check.",
                      f"Check which other variables are available for the rare categories of {a}.", None),
        "missing": (f"Is the missingness in {a} related to other variables?", "Missing-value chart and missingness patterns.",
                    "Look at which columns are missing together (Overview > missingness patterns).", None),
        "gap": (f"Why does {a} have a coverage gap, and does it affect trend analyses?", "Counts per period over time.",
                f"View observation counts per period for {a}.", dict(type="line_agg_count", x=a)),
    }[k]
    reason = {"corr": "A statistically reliable association with a non-trivial effect size was found.",
              "trend": "A monotonic drift over time was detected.", "group": "Groups differ by more than chance after correction.",
              "assoc": "Category mixes differ across groups beyond chance.", "skew": "Skew can mislead mean-based summaries.",
              "outlier": "Unusual values can dominate statistics.", "imbalance": "Rare categories limit what can be compared.",
              "missing": "Heavy missingness restricts every analysis using this column.", "gap": "Gaps undermine time-based conclusions."}[k]
    if not ins["passed"]:
        reason = "Exploratory only: this did NOT meet the reporting threshold after multiple-testing correction, so treat it as a hypothesis to check, not a finding."
    return dict(question=q, variables=v, reason=reason, analysis=an, evidence=ins["stat"], limitations=ins["limitations"],
                confounders=conf_txt, followup=fu, followup_viz=fviz, insight_id=ins["id"], kind=k)


def discover(df: pd.DataFrame, kinds: dict) -> dict:
    N = len(df)
    tests = run_tests(df, kinds) if N >= S.MIN_N else []
    passing = [t for t in tests if _passes(t)]
    ins_t = sorted((from_test(t, N, df) for t in passing), key=lambda i: -i["score"])
    ins_d = sorted(descriptive(df, kinds, N), key=lambda i: -i["score"])
    insights = ins_t + ins_d
    # questions: diverse, max 3 per kind, up to 7
    picked, counts = [], {}
    for i in insights:
        if len(picked) >= 7:
            break
        if counts.get(i["kind"], 0) < 3:
            picked.append(i)
            counts[i["kind"]] = counts.get(i["kind"], 0) + 1
    if len(picked) < 3:  # fall back to strongest untested-after-correction candidates, clearly labelled
        rest = sorted((t for t in tests if not _passes(t)), key=lambda t: t["p"])
        for t in rest[:3 - len(picked)]:
            picked.append(from_test(t, N, df))
    questions = [make_question(i, passing, df, kinds) for i in picked]
    return dict(insights=insights, questions=questions, n_tests=len(tests), n_passing=len(passing), methods=METHODS)
