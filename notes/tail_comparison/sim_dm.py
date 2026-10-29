#!/usr/bin/env python3
"""Small-sample corrections of the Diebold-Mariano test, main.tex Section 7.

Re-runs the null design of sim.py with the same seeds (so the plain DM rates
reproduce Table 2) and adds three corrections that are standard in the
forecasting literature, each two-sided at 5%:
  hln    Harvey, Leybourne and Newbold (1997): statistic scaled by
         sqrt((T + 1 - 2h + h(h-1)/T) / T), h = 1, with t_{T-1} critical values
  wpe    fixed-smoothing test of Coroneo and Iacone (2020): weighted periodogram
         variance on the m = floor(T^(1/3)) lowest Fourier frequencies,
         t_{2m} critical values
  boot   bootstrap-t: the outcomes are independent in this design, so pairs
         are resampled independently; 499 resamples, equal-tailed
For each, the share of replications declaring the narrower (resp. wider)
forecaster better.  Also the power of the bootstrap-t test in the detection
design of sim.py, at the same horizons and with the same samples.

Usage: python sim_dm.py [--reps 1000]
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

import cs
from sim import ALPHAS_D, ALPHAS_N, DELTA, DIST, POWER_MULT, T_LIST, gauss_pair_moments, null_pair

B_BOOT = 499
METHODS = ("dm", "hln", "wpe", "boot")


def wpe_stat(d):
    T = len(d)
    m = int(round(T ** (1 / 3)))
    m -= m ** 3 > T                                # exact floor of the cube root
    f = np.fft.rfft(d)[1:m + 1]
    lrv = np.mean(np.abs(f) ** 2) / T
    return d.mean() / np.sqrt(lrv / T), 2 * m


def boot_reject(d, rng):
    """Equal-tailed 5% bootstrap-t test of E d = 0; returns -1, 0 or +1."""
    T = len(d)
    m, s = d.mean(), d.std(ddof=1)
    if s == 0:
        return 0
    t0 = m / (s / np.sqrt(T))
    idx = rng.integers(0, T, size=(B_BOOT, T))
    x = d[idx]
    sb = x.std(axis=1, ddof=1)
    tb = (x.mean(axis=1) - m) / np.maximum(sb, 1e-300) * np.sqrt(T)
    tb[sb == 0] = 0.0
    lo, hi = np.quantile(tb, [0.025, 0.975])
    return -1 if t0 < lo else (1 if t0 > hi else 0)


def boot_reject_fast(d, c, rng):
    """boot_reject for d equal to the constant c except on a few rounds: a
    resample draws Binomial(T, k/T) of the k other rounds, uniformly, and c
    otherwise, which has the same law as an ordinary resample."""
    T = len(d)
    x = d[d != c] - c
    m, s = d.mean(), d.std(ddof=1)
    if s == 0:
        return 0
    t0 = m / (s / np.sqrt(T))
    n = rng.binomial(T, len(x) / T, size=B_BOOT)
    tb = np.zeros(B_BOOT)
    for i in range(B_BOOT):
        z = x[rng.integers(0, len(x), n[i])]
        s1, s2 = z.sum(), (z * z).sum()
        mb = c + s1 / T
        vb = (s2 - s1 * s1 / T) / (T - 1)
        tb[i] = (mb - m) / np.sqrt(vb / T) if vb > 0 else 0.0
    lo, hi = np.quantile(tb, [0.025, 0.975])
    return -1 if t0 < lo else (1 if t0 > hi else 0)


def run_detect(args):
    """Power at the horizons of sim.run_detect, on the same samples."""
    sign, alpha, reps, seed = args
    cA, cB, mean, var, H = gauss_pair_moments(alpha, sign)
    T_max = int(min(2e6, 25 * var / mean ** 2 * np.log(2 * cs.K_GRID / DELTA) + 2000))
    T80 = (1.96 + 0.8416) ** 2 * var / mean ** 2
    horizons = [int(np.ceil(m * T80)) for m in POWER_MULT[:2]]
    T_max = max(T_max, max(horizons))
    rng = np.random.default_rng(seed)
    brng = np.random.default_rng(seed + 10 ** 6)
    pw = {k: np.zeros(len(horizons)) for k in ("dm", "boot", "boot_wrong")}
    c = 2 * (cA - cB)
    for _ in range(reps):
        y = rng.standard_normal(T_max)
        d = cs.interval_score(-cA, cA, y, alpha) - cs.interval_score(-cB, cB, y, alpha)
        d[np.abs(y) <= min(cA, cB)] = c            # exact, to remove rounding noise
        for j, h in enumerate(horizons):          # A is worse: Delta > 0
            pw["dm"][j] += cs.dm_stat(d[:h]) > 1.96
            r = boot_reject_fast(d[:h], c, brng)
            pw["boot"][j] += r > 0
            pw["boot_wrong"][j] += r < 0
    return {"sign": sign, "alpha": alpha, "horizons": horizons, "reps": reps,
            **{f"power_{k}": (v / reps).tolist() for k, v in pw.items()}}


def run(args):
    design, alpha, reps, seed = args
    dist = DIST["gauss" if design == "gauss" else "t3"]
    cA, cB = null_pair(alpha, dist)
    rng = np.random.default_rng(seed)            # same stream as sim.run_null
    brng = np.random.default_rng(seed + 10 ** 6)
    Tm = max(T_LIST)
    out = {f"{k}_{side}": np.zeros(len(T_LIST)) for k in METHODS for side in ("narrow", "wide")}
    by_miss = []                                  # (misses of A by T=250, DM sign at 250)
    for _ in range(reps):
        if design == "gauss":
            s = np.ones(Tm)
            y = rng.standard_normal(Tm)
        else:
            s = np.exp(0.5 * rng.standard_normal(Tm))
            y = s * rng.standard_t(3, Tm)
        d = cs.interval_score(-cA * s, cA * s, y, alpha) - cs.interval_score(-cB * s, cB * s, y, alpha)
        for j, T in enumerate(T_LIST):
            x = d[:T]
            t = cs.dm_stat(x)
            sgn = {"dm": np.sign(t) * (abs(t) > 1.96)}
            th = t * np.sqrt((T - 1) / T)
            sgn["hln"] = np.sign(th) * (abs(th) > stats.t.isf(0.025, T - 1))
            tw, df = wpe_stat(x)
            sgn["wpe"] = np.sign(tw) * (abs(tw) > stats.t.isf(0.025, df))
            sgn["boot"] = boot_reject(x, brng)
            for k in METHODS:
                out[f"{k}_narrow"][j] += sgn[k] < 0
                out[f"{k}_wide"][j] += sgn[k] > 0
            if j == 0:
                by_miss.append((int(np.sum(np.abs(y[:T]) > cA * s[:T])), int(sgn["dm"])))
    res = {k: (v / reps).tolist() for k, v in out.items()}
    res.update(design=design, alpha=alpha, T=T_LIST, reps=reps, by_miss_250=by_miss)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--reps-detect", type=int, default=200)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(__file__).resolve().parent / args.out
    jobs = [(dsg, a, args.reps, 1000 + i) for i, (dsg, a) in
            enumerate((dsg, a) for dsg in ("gauss", "t3") for a in ALPHAS_N)]
    det_jobs = [(s, a, args.reps_detect, 5000 + i) for i, (s, a) in
                enumerate((s, a) for s in (-1, 1) for a in ALPHAS_D)]
    with ProcessPoolExecutor(max_workers=16) as ex:
        det = ex.map(run_detect, det_jobs)
        res = list(ex.map(run, jobs))
        det = list(det)
    (out / "sim_dm.json").write_text(json.dumps({"null": res, "detect": det}, indent=1))
    for r in det:
        print(f"{'narrow' if r['sign'] < 0 else 'widen':6s} a={r['alpha']:<6} power DM={r['power_dm']}"
              f" boot={r['power_boot']} boot wrong={r['power_boot_wrong']}")
    for r in res:
        print(f"{r['design']:5s} a={r['alpha']:<5}", "  ".join(
            f"{k}: {['%.1f' % (100 * v) for v in r[k + '_narrow']]}/{['%.1f' % (100 * v) for v in r[k + '_wide']]}"
            for k in METHODS))


if __name__ == "__main__":
    main()
