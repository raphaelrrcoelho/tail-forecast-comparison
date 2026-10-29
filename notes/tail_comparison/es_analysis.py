#!/usr/bin/env python3
"""Comparisons of (VaR, ES) forecasts by the FZ0 loss, main.tex Section 9.

For each index, level and pair of models (A, B), with d_t = FZ0(A) - FZ0(B):
  full sample   DM test (two-sided 5%, Newey-West) on all evaluation days;
                one-sided EB confidence bounds on the days where A's ES is the
                less conservative (certifies "A worse there" if positive) and
                on the days where B's is (certifies "B worse there");
  250-day       the same DM test in each non-overlapping 250-day window,
  windows       recording whether a significant result favours the model
                whose ES was less conservative on average in the window,
                and whether its sign agrees with the full-sample mean.
The EB bounds use the outcome floored at kappa times the more conservative
ES forecast (cs.fz0_floored), which keeps them valid without assumptions.
The first WARM evaluation days fix the lambda grid and are excluded.
Inputs come from es_forecasts.py.

Usage: python es_analysis.py [--kappa 2]
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

import cs
from es_forecasts import MODELS, TAUS, TICKERS

HERE = Path(__file__).resolve().parent
WARM, WIN, DELTA, Z = 250, 250, 0.05, 1.959964


def nw_lag(n):
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def one_sided(d, a, b, c, sel, lam0):
    """EB bounds on the selected days, at delta/2 per side (one side vacuous)."""
    if sel.sum() == 0:
        return -np.inf, np.inf
    L, U = cs.eb_cs(d[sel], a[sel], b[sel], c[sel], DELTA, lam0, idx=[int(sel.sum()) - 1])
    return float(L[0]), float(U[0])


def one_sided_tuned(d, a, b, c, sel):
    """Tuned EB bounds (cs.eb_test_split) on the selected days."""
    if sel.sum() < 20:
        return -np.inf, np.inf
    return cs.eb_test_split(d[sel], a[sel], b[sel], c[sel], DELTA)


def run(name, tau, kappa=2.0):
    z = np.load(HERE / "data" / f"es_{name.replace(' ', '_').replace('&', '')}.npz")
    r, s0 = z["r"], int(z["start"])
    fc = {m: z[f"{m}::{tau}"][s0:] for m in MODELS}
    y = r[s0:]
    rows = []
    for A, B in itertools.combinations(MODELS, 2):
        vA, eA = fc[A][:, 0], fc[A][:, 1]
        vB, eB = fc[B][:, 0], fc[B][:, 1]
        d = cs.fz0(vA, eA, y, tau) - cs.fz0(vB, eB, y, tau)
        a_less = eA >= eB                                   # A's ES less conservative
        ell = kappa * np.minimum(eA, eB)       # outcome floor
        dfl, a, b, c = cs.fz0_floored(vA, eA, vB, eB, y, tau, ell)
        lam0 = 1 / (2 * max(np.median((b - a)[:WARM]), 1e-8))
        ev = slice(WARM, None)
        d_, a_, b_, c_, al = d[ev], a[ev], b[ev], c[ev], a_less[ev]
        dfl_ = dfl[ev]
        T = len(d_)
        exc = (y[ev] <= np.maximum(vA[ev], vB[ev]))
        tstat = cs.dm_stat(d_, nw_lag(T))
        LA, _ = one_sided(dfl_, a_, b_, c_, al, lam0)        # days A less conservative
        _, UB = one_sided(dfl_, a_, b_, c_, ~al, lam0)       # days B less conservative
        LAt, _ = one_sided_tuned(dfl_, a_, b_, c_, al)
        _, UBt = one_sided_tuned(dfl_, a_, b_, c_, ~al)
        full = {"mean": float(d_.mean()), "dm_t": float(tstat), "dm": int(np.sign(tstat) * (abs(tstat) > Z)),
                "eb_A_worse": bool(LA > 0), "eb_B_worse": bool(UB < 0),
                "ebt_A_worse": bool(LAt > 0), "ebt_B_worse": bool(UBt < 0),
                "frac_A_less": float(al.mean()), "n_exc": int(exc.sum()),
                "mean_A_less": float(d_[al].mean()) if al.any() else float("nan"),
                "mean_B_less": float(d_[~al].mean()) if (~al).any() else float("nan")}
        wins = []
        for w0 in range(0, T - WIN + 1, WIN):
            sl = slice(w0, w0 + WIN)
            dw = d_[sl]
            tw = cs.dm_stat(dw, nw_lag(WIN))
            sig = int(np.sign(tw) * (abs(tw) > Z))
            rest = np.delete(d_, np.arange(w0, w0 + WIN))   # all other evaluation days
            # +1 if A was less conservative on average in the window
            less = 1 if eA[ev][sl].mean() > eB[ev][sl].mean() else -1
            LwA, _ = one_sided(dfl_[sl], a_[sl], b_[sl], c_[sl], al[sl], lam0)
            _, UwB = one_sided(dfl_[sl], a_[sl], b_[sl], c_[sl], ~al[sl], lam0)
            wins.append({"dm": sig, "n_exc": int(exc[sl].sum()), "eb": bool(LwA > 0 or UwB < 0),
                         # sig = -1 favours A (lower loss); favours less conservative if sig = -less
                         "favours_less": bool(sig != 0 and sig == -less),
                         "agrees_full": bool(sig != 0 and sig == np.sign(full["mean"])),
                         "agrees_rest": bool(sig != 0 and sig == np.sign(rest.mean()))})
        rows.append({"A": A, "B": B, "full": full, "windows": wins})
    return {"index": name, "tau": tau, "T": int(len(y) - WARM), "pairs": rows}


def summarize(results):
    summary = {}
    for tau in TAUS:
        R = [p for r in results if r["tau"] == tau for p in r["pairs"]]
        W = [w for p in R for w in p["windows"]]
        sigw = [w for w in W if w["dm"] != 0]
        fl = [w for w in sigw if w["favours_less"]]
        fm = [w for w in sigw if not w["favours_less"]]
        s = {"pairs": len(R),
             "full_dm_sig": float(np.mean([p["full"]["dm"] != 0 for p in R])),
             "full_eb_any": float(np.mean([p["full"]["eb_A_worse"] or p["full"]["eb_B_worse"] for p in R])),
             "full_ebt_any": float(np.mean([p["full"]["ebt_A_worse"] or p["full"]["ebt_B_worse"] for p in R])),
             "full_exc_med": float(np.median([p["full"]["n_exc"] for p in R])),
             "windows": len(W), "win_exc_med": float(np.median([w["n_exc"] for w in W])),
             "win_dm_sig": len(sigw) / len(W),
             "win_eb_any": float(np.mean([w["eb"] for w in W])),
             "win_sig_favours_less": len(fl) / max(len(sigw), 1),
             "win_wrong_sign_less": 1 - np.mean([w["agrees_full"] for w in fl]) if fl else float("nan"),
             "win_wrong_sign_more": 1 - np.mean([w["agrees_full"] for w in fm]) if fm else float("nan"),
             "win_wrong_sign_less_rest": 1 - np.mean([w["agrees_rest"] for w in fl]) if fl else float("nan"),
             "win_wrong_sign_more_rest": 1 - np.mean([w["agrees_rest"] for w in fm]) if fm else float("nan"),
             "full_sig_favours_less": float(np.mean([p["full"]["dm"] == -(1 if p["full"]["frac_A_less"] > 0.5 else -1)
                                                     for p in R if p["full"]["dm"] != 0]))}
        summary[str(tau)] = s
        print(f"tau={tau}: pairs={s['pairs']} full: exc med {s['full_exc_med']:.0f}, DM sig {s['full_dm_sig']:.2f},"
              f" EB certifies {s['full_eb_any']:.2f} | windows={s['windows']} exc med {s['win_exc_med']:.0f}:"
              f" DM sig {s['win_dm_sig']:.3f}, EB {s['win_eb_any']:.3f}, of which favour less conservative {s['win_sig_favours_less']:.2f};"
              f" wrong sign: favouring less {s['win_wrong_sign_less']:.2f}, favouring more {s['win_wrong_sign_more']:.2f}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=float, default=2.0,
                    help="outcome floor as a multiple of the more conservative ES")
    args = ap.parse_args()
    results = [run(n, tau, args.kappa) for tau in TAUS for n in TICKERS]
    summary = summarize(results)
    tag = "" if args.kappa == 2.0 else f"_kappa{args.kappa:g}"
    (HERE / "results" / f"es{tag}.json").write_text(
        json.dumps({"kappa": args.kappa, "results": results, "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
