#!/usr/bin/env python3
"""Sampling uncertainty of the window results in main.tex Section 9.

The 2130 window-pair tests per level are dependent: the 15 pairs of an index
share the window's returns, and windows of different indices overlap in
calendar time.  We therefore resample calendar years: each window is assigned
the year of its midpoint, and a bootstrap draws years with replacement and
keeps every window (all indices and pairs) of the drawn years.  Reports 95%
percentile intervals for the share of significant results favouring the less
conservative model and for the wrong-sign rates of each kind (judged against
all other days), and for the difference of the two wrong-sign rates.

Usage: python es_uncertainty.py   (after es_forecasts.py and es_analysis.py)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from es_analysis import WARM, WIN
from es_forecasts import TAUS, WINDOW

HERE = Path(__file__).resolve().parent
B = 4000


def window_years(name, n_windows):
    dates = pd.read_csv(HERE / "data" / "returns.csv", index_col=0, parse_dates=True)[name].dropna().index
    s = WINDOW + WARM
    return [dates[s + w * WIN + WIN // 2].year for w in range(n_windows)]


def rates(fl_wrong, fm_wrong, fav_less):
    a, b = np.mean(fl_wrong), np.mean(fm_wrong)
    return {"favours_less": np.mean(fav_less), "wrong_less": a, "wrong_more": b, "diff": a - b}


def main():
    res = json.loads((HERE / "results" / "es.json").read_text())["results"]
    rng = np.random.default_rng(7)
    out = {}
    for tau in TAUS:
        rows = []                                  # (year, favours_less, wrong)
        for r in res:
            if r["tau"] != tau:
                continue
            yrs = window_years(r["index"], len(r["pairs"][0]["windows"]))
            for p in r["pairs"]:
                for y, w in zip(yrs, p["windows"]):
                    if w["dm"] != 0:
                        rows.append((y, w["favours_less"], not w["agrees_rest"]))
        yr = np.array([x[0] for x in rows])
        fl = np.array([x[1] for x in rows])
        wr = np.array([x[2] for x in rows])
        years = np.unique(yr)
        groups = [np.nonzero(yr == y)[0] for y in years]
        point = rates(wr[fl], wr[~fl], fl)
        boot = {k: [] for k in point}
        for _ in range(B):
            ix = np.concatenate([groups[i] for i in rng.integers(0, len(years), len(years))])
            f, w = fl[ix], wr[ix]
            if f.all() or (~f).all():
                continue
            for k, v in rates(w[f], w[~f], f).items():
                boot[k].append(v)
        out[str(tau)] = {"years": len(years), "significant": int(len(rows)),
                         "point": {k: float(v) for k, v in point.items()},
                         "ci95": {k: [float(q) for q in np.quantile(v, [0.025, 0.975])] for k, v in boot.items()},
                         "p_diff_le_0": float(np.mean(np.array(boot["diff"]) <= 0))}
        o = out[str(tau)]
        print(f"tau={tau}: {o['years']} years, {o['significant']} significant;",
              ", ".join(f"{k} {100 * o['point'][k]:.1f} [{100 * o['ci95'][k][0]:.1f}, {100 * o['ci95'][k][1]:.1f}]"
                        for k in point), f"P(diff<=0)={o['p_diff_le_0']:.3f}")
    (HERE / "results" / "es_uncertainty.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
