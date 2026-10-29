#!/usr/bin/env python3
"""Do the rate conditions of Theorem 2 hold in the online conformal data?

For each level, over the methods selected by empirical.py:
  miss/alpha     realized miss rate of each method, relative to alpha
  rolling        90th percentile, over windows of 20/alpha rounds, of the
                 miss rate relative to alpha (a proxy for the conditional C)
  Delta/H        |mean score difference| / mean endpoint distance, per pair
  H q90/median   spread of the endpoint distance (scaled by sigma_t)
  oracle         share of pairs separated by the empirical-Bernstein bound
                 with the single lambda chosen in hindsight (not a valid
                 procedure: an upper limit on what tuning the grid can gain)
Inputs: results/empirical.json and data/*.npz.  Output: results/conditions.json.

Usage: python conditions.py
"""
import json
from pathlib import Path

import numpy as np

import cs
from empirical import ALPHAS, DELTA, load

LAMS = np.exp(np.linspace(-25, 5, 400))


def oracle_separates(d, a, b, c):
    """|mean| exceeds the fixed-lambda EB radius at the best lambda in hindsight."""
    m, T = d.mean(), len(d)
    R = (c - a) if m > 0 else (b - c)
    D2 = (d - c) ** 2
    best = np.inf
    for lam in LAMS:
        u = lam * R
        if np.any(u >= 1):
            continue
        best = min(best, (np.log(2 / DELTA) + lam ** 2 * np.sum(cs._psi_ratio(u) * D2)) / (lam * T))
    return abs(m) > best

HERE = Path(__file__).resolve().parent


def main():
    d = json.loads((HERE / "results" / "empirical.json").read_text())
    out = {}
    for alpha in ALPHAS:
        R = [r for r in d["results"] if r["alpha"] == alpha]
        P = [p for r in R for p in r["pairs"]]
        miss = [(1 - p[k]) / alpha for p in P for k in ("cov_A", "cov_B")]
        eff = [abs(p["full"]["mean"]) / p["H_mean_raw"] for p in P]
        spread = [p["H_q90"] / max(p["H_med"], 1e-12) for p in P]
        roll, orc = [], []
        for r in R:
            y, _, iv, tr = load(r["dataset"], alpha)
            for p in r["pairs"]:
                IA, IB = iv[r["reps"][p["A"]]], iv[r["reps"][p["B"]]]
                dd = cs.interval_score(IA[:, 0], IA[:, 1], y, alpha) - cs.interval_score(IB[:, 0], IB[:, 1], y, alpha)
                a, b, c = cs.interval_range(IA[:, 0], IA[:, 1], IB[:, 0], IB[:, 1], alpha)
                orc.append(oracle_separates(dd[tr:], a[tr:], b[tr:], c[tr:]))
            for cfg in r["reps"].values():
                I, yy = iv[cfg][tr:], y[tr:]
                m = ((yy < I[:, 0]) | (yy > I[:, 1])).astype(float)
                w = int(round(20 / alpha))
                k = len(m) // w
                roll.append(float(np.quantile(m[:k * w].reshape(k, w).mean(1) / alpha, 0.9)))
        s = {"miss_med": float(np.median(miss)), "miss_min": float(np.min(miss)),
             "miss_max": float(np.max(miss)), "roll_med": float(np.median(roll)),
             "roll_max": float(np.max(roll)), "eff_med": float(np.median(eff)),
             "eff_q25": float(np.quantile(eff, 0.25)), "eff_q75": float(np.quantile(eff, 0.75)),
             "spread_med": float(np.median(spread)), "oracle_sig": float(np.mean(orc))}
        out[str(alpha)] = s
        print(alpha, {k: round(v, 2) for k, v in s.items()})
    (HERE / "results" / "conditions.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
