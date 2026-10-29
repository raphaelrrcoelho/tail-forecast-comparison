#!/usr/bin/env python3
"""Simulations for main.tex, Section 7.

Design N (null): forecaster A issues a narrow interval [-cA s_t, cA s_t] and
B a wide one [-cB s_t, cB s_t], with cA = 0.85 q_{1-alpha/2} and cB chosen so
that both have the same expected interval score; Delta_bar_t = 0.  Outcomes
are Gaussian with s_t = 1, or Student-t(3) with a lognormal scale s_t known
to both forecasters.  Reports, over R replications:
  dm_narrow / dm_wide  two-sided 5% DM test declares A (resp. B) better at T
  eb / hoef / asym     the 95% confidence sequence excludes 0 at some t <= T
                       (time-uniform error), and at t = T (fixed-T error)

Design D (detection): Gaussian outcomes; B is the exact central interval,
A is narrowed (or widened) by eta / z, so A is worse: Delta > 0.  Reports
the first round at which the EB lower confidence bound exceeds 0, and the
round at which the (deterministic) Hoeffding radius falls below Delta.

Usage: python sim.py [--reps 1000] [--out results]
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.integrate import quad
from scipy.optimize import brentq

import cs

ALPHAS_N = [0.2, 0.1, 0.05, 0.02, 0.01]
T_LIST = [250, 1000, 4000]
ALPHAS_D = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
ETA, DELTA = 0.5, 0.05
POWER_MULT = [1, 2, 4]   # horizons, in multiples of the known-variance 80%-power sample size
DIST = {"gauss": stats.norm, "t3": stats.t(3)}


def expected_is(c, alpha, dist):
    over = 2 * quad(lambda s: dist.sf(c + s), 0, np.inf, limit=200)[0]
    return 2 * c + 2 / alpha * over


def null_pair(alpha, dist):
    q = dist.isf(alpha / 2)
    cA = 0.85 * q
    target = expected_is(cA, alpha, dist)
    cB = brentq(lambda c: expected_is(c, alpha, dist) - target, q, 20 * q)
    return cA, cB


def run_null(args):
    design, alpha, reps, seed = args
    dist = DIST["gauss" if design == "gauss" else "t3"]
    cA, cB = null_pair(alpha, dist)
    rng = np.random.default_rng(seed)
    Tm = max(T_LIST)
    out = {k: np.zeros(len(T_LIST)) for k in
           ("dm_narrow", "dm_wide", "eb_u", "eb_f", "ebt_f", "hoef_u", "hoef_f", "asym_u", "asym_f")}
    misses = np.zeros(len(T_LIST))
    for _ in range(reps):
        if design == "gauss":
            s = np.ones(Tm)
            y = rng.standard_normal(Tm)
        else:
            s = np.exp(0.5 * rng.standard_normal(Tm))
            y = s * rng.standard_t(3, Tm)
        d = cs.interval_score(-cA * s, cA * s, y, alpha) - cs.interval_score(-cB * s, cB * s, y, alpha)
        a, b, c = cs.interval_range(-cA * s, cA * s, -cB * s, cB * s, alpha)
        lam0 = 1 / (2 * (b[0] - a[0]))
        L, U = cs.eb_cs(d, a, b, c, DELTA, lam0)
        Lh, Uh = cs.hoeffding_cs(d, a, b, DELTA, lam0)
        La, Ua = cs.asymp_cs(d, DELTA)
        for j, T in enumerate(T_LIST):
            tstat = cs.dm_stat(d[:T])
            out["dm_narrow"][j] += tstat < -1.96
            out["dm_wide"][j] += tstat > 1.96
            for key, (lo, hi) in (("eb", (L, U)), ("hoef", (Lh, Uh)), ("asym", (La, Ua))):
                start = 0 if key != "asym" else 30
                out[f"{key}_u"][j] += np.any((lo[start:T] > 0) | (hi[start:T] < 0))
                out[f"{key}_f"][j] += (lo[T - 1] > 0) or (hi[T - 1] < 0)
            Lt, Ut = cs.eb_test_split(d[:T], a[:T], b[:T], c[:T], DELTA)
            out["ebt_f"][j] += (Lt > 0) or (Ut < 0)
            misses[j] += np.sum(np.abs(y[:T]) > cA * s[:T])
    res = {k: (v / reps).tolist() for k, v in out.items()}
    res.update(design=design, alpha=alpha, cA=cA, cB=cB, T=T_LIST,
               mean_misses_A=(misses / reps).tolist(), reps=reps)
    return res


def gauss_pair_moments(alpha, sign):
    """Exact mean, variance and endpoint distance of d = IS(A) - IS(B) for
    B = [-z, z] and A = [-(z + sign r), z + sign r], r = ETA / z."""
    z = stats.norm.isf(alpha / 2)
    cA, cB = z + sign * ETA / z, z
    lo, hi = min(cA, cB), max(cA, cB)
    # d = 2(cA - cB) + (2/alpha) f, f = sign_f * min((|Y| - lo)_+, hi - lo)
    sf = -1.0 if cA > cB else 1.0          # f >= 0 iff A is the narrower one
    m1 = 2 * quad(lambda u: stats.norm.sf(lo + u), 0, hi - lo)[0]
    m2 = 4 * quad(lambda u: u * stats.norm.sf(lo + u), 0, hi - lo)[0]
    mean = 2 * (cA - cB) + 2 / alpha * sf * m1
    var = 4 / alpha ** 2 * (m2 - m1 ** 2)
    return cA, cB, mean, var, hi - lo


def run_detect(args):
    sign, alpha, reps, seed = args
    cA, cB, mean, var, H = gauss_pair_moments(alpha, sign)
    T_max = int(min(2e6, 25 * var / mean ** 2 * np.log(2 * cs.K_GRID / DELTA) + 2000))
    idx = np.unique(np.geomspace(1, T_max, 600).astype(int)) - 1
    rng = np.random.default_rng(seed)
    T80 = (1.96 + 0.8416) ** 2 * var / mean ** 2
    horizons = [int(np.ceil(m * T80)) for m in POWER_MULT]
    T_max = max(T_max, max(horizons))
    pw = {k: np.zeros(len(horizons)) for k in ("dm", "dm_wrong", "ebt")}
    times, misses = [], []
    for _ in range(reps):
        y = rng.standard_normal(T_max)
        d = cs.interval_score(-cA, cA, y, alpha) - cs.interval_score(-cB, cB, y, alpha)
        ones = np.ones(T_max)
        a, b, c = cs.interval_range(-cA * ones, cA * ones, -cB * ones, cB * ones, alpha)
        lam0 = 1 / (2 * (b[0] - a[0]))
        L, _ = cs.eb_cs(d, a, b, c, DELTA, lam0, idx=idx)
        for j, h in enumerate(horizons):          # A is worse: Delta > 0
            tstat = cs.dm_stat(d[:h])
            pw["dm"][j] += tstat > 1.96
            pw["dm_wrong"][j] += tstat < -1.96
            pw["ebt"][j] += cs.eb_test_split(d[:h], a[:h], b[:h], c[:h], DELTA)[0] > 0
        hit = np.nonzero(L > 0)[0]
        t = idx[hit[0]] + 1 if len(hit) else np.inf
        times.append(t)
        misses.append(np.sum(np.abs(y[:int(t)]) > min(cA, cB)) if np.isfinite(t) else np.nan)
    # Hoeffding: L_t = mean_t - rho_t with rho_t deterministic (constant ranges)
    a0, b0 = float(a[0]), float(b[0])
    lam = cs.lambda_grid(1 / (2 * (b0 - a0)), K=2 * cs.K_GRID)
    t_grid = np.unique(np.geomspace(1, 1e9, 2000).astype(np.int64))
    pen = lam ** 2 * (b0 - a0) ** 2 / 8
    logd = np.log(2 / DELTA)
    rho = np.empty(len(t_grid))
    for i, t in enumerate(t_grid):
        f = lambda u: np.logaddexp.reduce(-np.log(len(lam)) + lam * t * u - pen * t) - logd
        rho[i] = brentq(f, 0, 1e6 * (b0 - a0))
    below = np.nonzero(rho <= mean)[0]
    t_hoef = int(t_grid[below[0]]) if len(below) else None
    times = np.array(times, dtype=float)
    return {"sign": sign, "alpha": alpha, "cA": cA, "cB": cB, "Delta": mean, "var": var,
            "H": H, "T_max": T_max, "reps": reps,
            "eb_median": float(np.median(times)), "eb_q10": float(np.quantile(times, 0.1)),
            "eb_q90": float(np.quantile(times, 0.9)), "eb_censored": float(np.mean(~np.isfinite(times))),
            "eb_misses_median": float(np.nanmedian(misses)),
            "hoeffding": t_hoef,
            "fixed_T_80": float(T80), "power_horizons": horizons,
            "power_dm": (pw["dm"] / reps).tolist(), "power_dm_wrong": (pw["dm_wrong"] / reps).tolist(),
            "power_ebt": (pw["ebt"] / reps).tolist(),
            "lower_scale": float(H ** 2 / (alpha * mean ** 2))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--reps-detect", type=int, default=200)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(__file__).resolve().parent / args.out
    out.mkdir(exist_ok=True)
    null_jobs = [(dsg, a, args.reps, 1000 + i) for i, (dsg, a) in
                 enumerate((dsg, a) for dsg in ("gauss", "t3") for a in ALPHAS_N)]
    det_jobs = [(s, a, args.reps_detect, 5000 + i) for i, (s, a) in
                enumerate((s, a) for s in (-1, 1) for a in ALPHAS_D)]
    with ProcessPoolExecutor(max_workers=16) as ex:
        det = ex.map(run_detect, det_jobs)
        null = ex.map(run_null, null_jobs)
        null, det = list(null), list(det)
    (out / "sim_null.json").write_text(json.dumps(null, indent=1))
    (out / "sim_detect.json").write_text(json.dumps(det, indent=1))
    for r in null:
        print(f"{r['design']:5s} a={r['alpha']:<5} misses(A)={['%.1f' % m for m in r['mean_misses_A']]}"
              f" DMnarrow={r['dm_narrow']} DMwide={r['dm_wide']} EBu={r['eb_u']} EBt={r['ebt_f']} Hu={r['hoef_u']} ASu={r['asym_u']}")
    for r in det:
        print(f"{'narrow' if r['sign'] < 0 else 'widen':6s} a={r['alpha']:<6} Delta={r['Delta']:.4f} H={r['H']:.3f}"
              f" EB median={r['eb_median']:.0f} (misses {r['eb_misses_median']:.0f}) cens={r['eb_censored']:.2f}"
              f" Hoeffding={r['hoeffding']} fixedT80={r['fixed_T_80']:.0f} H2/(aD2)={r['lower_scale']:.0f}"
              f" power DM={r['power_dm']} EBt={r['power_ebt']}")


if __name__ == "__main__":
    main()
