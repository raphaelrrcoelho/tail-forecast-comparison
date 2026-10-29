#!/usr/bin/env python3
"""Online conformal comparisons for main.tex, Section 8.

For each dataset and level, take the best configuration of each online
conformal family (lowest mean interval score over the evaluation rounds)
and compare every pair of families.  For each pair report
  H        median endpoint distance H_t / sigma_t
  n_inf    rounds on which d_t differs from its centre c_t (a miss of
           either interval), and that count divided by T
  DM       two-sided 5% Diebold-Mariano test, Newey-West lag as in the
           earlier experiments (max(h - 1, floor(4 (T/100)^(2/9))))
  EB, Hoef 95% confidence sequences at the last round
  split    the same procedures on the first half, checked against the
           point estimate on the second half
Data come from extract_intervals.py.

Usage: python empirical.py [--out results]
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

import cs

HERE = Path(__file__).resolve().parent
DATASETS = ["ETTh1", "ETTm1", "Traffic", "ExchangeRate", "EPEX_SPOT", "M4_Hourly", "NN5_Daily"]
HORIZON = {"ETTh1": 24, "ETTm1": 96, "Traffic": 24, "ExchangeRate": 7,
           "EPEX_SPOT": 24, "M4_Hourly": 24, "NN5_Daily": 7}
FAMILIES = {"AgACI": "AgACI-", "DtACI": "DtACI-", "SAOCP": "SAOCP", "QT": "QT-",
            "PI": "PI-", "AgACI-W": "P1 AgACI-W"}
ALPHAS = [0.20, 0.10, 0.05, 0.01]
DELTA, Z = 0.05, 1.959964


def load(name, alpha):
    tag = f"{alpha:.2f}".replace(".", "")
    z = np.load(HERE / "data" / f"{name}_a{tag}.npz")
    tr = int(z["train"])
    iv = {k[4:]: z[k] for k in z.files if k.startswith("iv::")}
    return z["y"], z["sigma"], iv, tr


def decide(lo, hi):
    return 1 if lo > 0 else (-1 if hi < 0 else 0)


def compare(d, a, b, c, lam0, lag):
    L, U = cs.eb_cs(d, a, b, c, DELTA, lam0, idx=[len(d) - 1])
    Lh, Uh = cs.hoeffding_cs(d, a, b, DELTA, lam0, idx=[len(d) - 1])
    t = cs.dm_stat(d, lag)
    Lt, Ut = cs.eb_test_split(d, a, b, c, DELTA)
    return {"mean": float(d.mean()), "dm_t": float(t), "dm": int(np.sign(t) * (abs(t) > Z)),
            "eb": decide(L[0], U[0]), "eb_lo": float(L[0]), "eb_hi": float(U[0]),
            "ebt": decide(Lt, Ut),
            "hoef": decide(Lh[0], Uh[0])}


def run(name, alpha):
    y, sig, iv, tr = load(name, alpha)
    ev = slice(tr, None)
    reps = {}
    for fam, prefix in FAMILIES.items():
        keys = [k for k in iv if k.startswith(prefix)]
        scores = {k: cs.interval_score(iv[k][ev, 0], iv[k][ev, 1], y[ev], alpha).mean() for k in keys}
        reps[fam] = min(scores, key=scores.get)
    T = len(y) - tr
    lag = max(HORIZON[name] - 1, int(np.floor(4 * (T / 100) ** (2 / 9))))
    rows = []
    for A, B in itertools.combinations(FAMILIES, 2):
        IA, IB = iv[reps[A]], iv[reps[B]]
        lo_a, hi_a, lo_b, hi_b = IA[:, 0], IA[:, 1], IB[:, 0], IB[:, 1]
        d_all = cs.interval_score(lo_a, hi_a, y, alpha) - cs.interval_score(lo_b, hi_b, y, alpha)
        a_all, b_all, c_all = cs.interval_range(lo_a, hi_a, lo_b, hi_b, alpha)
        # grid scale from the warm-up rounds, which precede every evaluated round
        warm = np.median(b_all[:tr] - a_all[:tr])
        lam0 = 1 / (2 * max(warm, 1e-12))
        d, a, b, c = d_all[ev], a_all[ev], b_all[ev], c_all[ev]
        H = np.maximum(np.abs(lo_a - lo_b), np.abs(hi_a - hi_b))[ev] / sig[ev]
        inf = np.abs(d - c) > 1e-12 * np.maximum(1, np.abs(c))
        half = T // 2
        full = compare(d, a, b, c, lam0, lag)
        first = compare(d[:half], a[:half], b[:half], c[:half], lam0, lag)
        second = compare(d[half:], a[half:], b[half:], c[half:], lam0, lag)
        rows.append({"A": A, "B": B, "A_cfg": reps[A], "B_cfg": reps[B],
                     "cov_A": float(np.mean((lo_a[ev] <= y[ev]) & (y[ev] <= hi_a[ev]))),
                     "cov_B": float(np.mean((lo_b[ev] <= y[ev]) & (y[ev] <= hi_b[ev]))),
                     "H_med": float(np.median(H)), "H_q90": float(np.quantile(H, 0.9)),
                     "H_mean_raw": float(np.mean(np.maximum(np.abs(lo_a - lo_b), np.abs(hi_a - hi_b))[ev])),
                     "n_inf": int(inf.sum()), "frac_inf": float(inf.mean()),
                     "full": full, "first": first, "second": second})
    return {"dataset": name, "alpha": alpha, "T": T, "lag": lag, "reps": reps, "pairs": rows}


def summarize(results):
    print(f"{'alpha':>6s} {'pairs':>5s} {'n_inf/T/alpha':>13s} {'n_inf med':>9s} "
          f"{'DM sig':>6s} {'EB sig':>6s} {'Hoef sig':>8s} {'DM&!EB':>6s} "
          f"{'DM1 sig':>7s} {'DM1 rep':>7s} {'EB1 sig':>7s} {'EB1 rep':>7s} {'need':>7s} {'enough':>6s}")
    summary = {}
    for alpha in ALPHAS:
        P = [p for r in results if r["alpha"] == alpha for p in r["pairs"]]
        if not P:
            continue
        T = {r["dataset"]: r["T"] for r in results if r["alpha"] == alpha}
        f = lambda k, s="full": np.mean([p[s][k] != 0 for p in P])
        dm1 = [p for p in P if p["first"]["dm"] != 0]
        eb1 = [p for p in P if p["first"]["eb"] != 0]
        ebt1 = [p for p in P if p["first"]["ebt"] != 0]
        rep = lambda S, k: (np.mean([np.sign(p["second"]["mean"]) == p["first"][k] for p in S])
                            if S else float("nan"))
        s = {"pairs": len(P), "frac_inf_over_alpha": float(np.median([p["frac_inf"] / alpha for p in P])),
             "n_inf_med": float(np.median([p["n_inf"] for p in P])),
             "dm_sig": float(f("dm")), "eb_sig": float(f("eb")), "hoef_sig": float(f("hoef")),
             "dm_not_eb": float(np.mean([p["full"]["dm"] != 0 and p["full"]["eb"] == 0 for p in P])),
             "dm_eb_conflict": int(sum(p["full"]["dm"] * p["full"]["eb"] < 0 for p in P)),
             "dm1_sig": len(dm1) / len(P), "dm1_rep": float(rep(dm1, "dm")),
             "eb1_sig": len(eb1) / len(P), "eb1_rep": float(rep(eb1, "eb")),
             "ebt_sig": float(f("ebt")), "ebt1_sig": len(ebt1) / len(P), "ebt1_rep": float(rep(ebt1, "ebt")),
             "ebt1_n": len(ebt1), "dm1_n": len(dm1), "eb1_n": len(eb1),
             "dm_ebt_conflict": int(sum(p["full"]["dm"] * p["full"]["ebt"] < 0 for p in P)),
             # misses needed by the rate (H/Delta)^2 log(1/delta), alpha-scaled units
             "misses_needed_med": float(np.median([(p["H_mean_raw"] / abs(p["full"]["mean"])) ** 2
                                                   * np.log(1 / DELTA) for p in P])),
             "frac_enough_misses": float(np.mean([p["n_inf"] >= (p["H_mean_raw"] / abs(p["full"]["mean"])) ** 2
                                                  * np.log(1 / DELTA) for p in P]))}
        summary[str(alpha)] = s
        print(f"{alpha:6.2f} {s['pairs']:5d} {s['frac_inf_over_alpha']:13.2f} {s['n_inf_med']:9.0f} "
              f"{s['dm_sig']:6.2f} {s['eb_sig']:6.2f} {s['hoef_sig']:8.2f} {s['dm_not_eb']:6.2f} "
              f"{s['dm1_sig']:7.2f} {s['dm1_rep']:7.2f} {s['eb1_sig']:7.2f} {s['eb1_rep']:7.2f} "
              f"{s['misses_needed_med']:7.0f} {s['frac_enough_misses']:6.2f}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    args = ap.parse_args()
    results = []
    for alpha in args.alphas:
        for name in DATASETS:
            tag = f"{alpha:.2f}".replace(".", "")
            if not (HERE / "data" / f"{name}_a{tag}.npz").exists():
                print(f"missing {name} alpha={alpha}")
                continue
            results.append(run(name, alpha))
    out = HERE / args.out
    out.mkdir(exist_ok=True)
    summary = summarize(results)
    (out / "empirical.json").write_text(json.dumps({"results": results, "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
