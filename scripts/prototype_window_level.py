#!/usr/bin/env python3
"""Go/no-go prototype: choose the calibration window and the level jointly.

Question: does combining window choice with level adaptation beat the
strongest level-adaptive baselines, AgACI and DtACI, each at its best
reference window in hindsight?  Every method runs on the matched protocol:
one point forecast, one local scale sigma_t and one stream of standardized
conformity scores s_t = |y_t - mu_t| / sigma_t, so only the adaptation rule
differs.

Experts are ACI runs indexed by (window n, step size gamma); gamma = 0 is a
fixed level, so the gamma = 0 experts are exactly the CIS-TS candidates.

  P1  AgACI-W  ML-Poly aggregation of all experts' lower and upper bounds
               under the pinball loss: AgACI extended across windows.
  P2  CTS-W    contextual linear Thompson sampling over all experts with
               full-information updates: the CIS-TS selector whose arms
               adapt their level.
  CTS-5        the same selector over the gamma = 0 experts only (CIS-TS).

Pre-registered bar: P1 or P2 beats both baselines at their best window,
significantly (two-sided DM, 5%), on at least 5 of 7 datasets at
alpha = 0.10 and at alpha = 0.01, with coverage at most 0.5 points below
nominal.

Usage:
  python scripts/prototype_window_level.py --alpha 0.10 --output results/prototype_window_level
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conformal_ts.data.real_datasets import RealDatasetLoader  # noqa: E402
from conformal_ts.evaluation.metrics import diebold_mariano_test  # noqa: E402
from conformal_ts.experiments.bandit_experiment import (  # noqa: E402
    ACI_FAMILY_GAMMAS, _AgACI, _DtACI, _MLPoly, _RunningNormalizer,
    _conformal_quantile, _covering_level, build_scores_matrix_with_cqr)
from conformal_ts.forecasters import make_default_forecasters  # noqa: E402
from conformal_ts.theory import polylog_lambda_grid  # noqa: E402

DATASETS = ["ETTh1", "ETTm1", "Traffic", "ExchangeRate", "EPEX_SPOT",
            "M4_Hourly", "NN5_Daily"]
GRIDS = {0.10: [25, 50, 100, 200, 500], 0.01: [100, 200, 500, 1000, 2000]}
TRAIN = 50          # warm-up rounds, excluded from every reported number
SLIDE = 100         # sliding window of the Thompson-sampling posterior


def prepare(name, alpha, windows):
    """Series, point forecast, local scale and 15-d context, as in
    scripts/run_diagnostic.py."""
    raw = RealDatasetLoader().prepare_raw_series(name, num_specs=5)
    target, warmup, df, cfg = raw["target"], raw["warmup"], raw["df"], raw["config"]
    sp = cfg.seasonal_period or 24
    fcs = make_default_forecasters(seasonal_period=sp, rolling_window=50,
                                   trend_window=20, horizon=sp)
    mh = max(warmup, max(f.min_history for f in fcs), max(windows))
    S, _, y, iv, _, sigma = build_scores_matrix_with_cqr(
        target, forecasters=fcs, alpha=alpha, min_history=mh,
        calibration_windows=list(windows), return_local_scale=True)
    T = S.shape[0]
    ctx = RealDatasetLoader.build_context_features(df, target, mh, T, cfg)
    ref = min(range(len(windows)), key=lambda k: abs(windows[k] - 100))
    covered = ((iv[:, ref, 0] <= y) & (y <= iv[:, ref, 1])).astype(float)
    drift = []
    for x in (covered, S[:, ref]):
        c = np.concatenate([[0.0], np.cumsum(x)])
        lo = np.maximum(np.arange(T) - 100, 0)
        n = np.arange(T) - lo
        m = np.where(n > 0, (c[np.arange(T)] - c[lo]) / np.maximum(n, 1), 0.0)
        sd = m.std()
        drift.append((m - m.mean()) / sd if sd > 1e-8 else m)
    ctx = np.column_stack([ctx] + drift)
    mu = 0.5 * (iv[:, 0, 0] + iv[:, 0, 1])
    return y, mu, sigma, ctx, S, int(sp)


def interval_score(lo, hi, y, alpha):
    return (hi - lo) + (2 / alpha) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))


class LeanTS:
    """Full-information linear Thompson sampling with a shared design.

    Every arm is updated with the same context each round, so all arms
    share one precision matrix; the posterior of arm k is
    N(A^{-1} b_k, v^2 A^{-1}) over the last ``slide`` rounds.  Rewards are
    running-normalized and clipped at +-3, as in the CIS-TS runner.
    """

    def __init__(self, n_arms, dim, lam, seed, slide=SLIDE, v2=1.0):
        self.K, self.d, self.lam, self.v = n_arms, dim, lam, np.sqrt(v2)
        self.rng = np.random.default_rng(seed)
        self.X = np.zeros((slide, dim))
        self.R = np.zeros((slide, n_arms))
        self.n = 0
        self.slide = slide
        self.norm = _RunningNormalizer()

    def select(self, x):
        m = min(self.n, self.slide)
        X, R = self.X[:m], self.R[:m]
        A = self.lam * np.eye(self.d) + X.T @ X
        Ainv = np.linalg.inv(A)
        theta = Ainv @ (X.T @ R)
        L = np.linalg.cholesky(Ainv)
        draw = theta + self.v * (L @ self.rng.standard_normal((self.d, self.K)))
        return int(np.argmax(x @ draw))

    def update(self, x, raw_rewards):
        z = np.clip([self.norm.normalize(r) for r in raw_rewards], -3.0, 3.0)
        for r in raw_rewards:
            self.norm.update(r)
        i = self.n % self.slide
        self.X[i], self.R[i] = x, z
        self.n += 1


def run_selector(is_matrix, ctx, seed, lams):
    """Choose lambda by warm-up CV on the first TRAIN rounds, then run.

    Mirrors the CIS-TS runner: CV runs play round-robin for the first K
    rounds and score the rest of the prefix; the main run plays
    round-robin through the whole warm-up.  Returns the expert played at
    each round.
    """
    T, K = is_matrix.shape

    def play(lam, horizon, round_robin):
        ts = LeanTS(K, ctx.shape[1], lam, seed)
        picks = np.zeros(horizon, dtype=int)
        for t in range(horizon):
            picks[t] = t % K if t < round_robin else ts.select(ctx[t])
            ts.update(ctx[t], -is_matrix[t])
        return picks

    cv = []
    for lam in lams:
        played = is_matrix[np.arange(TRAIN), play(lam, TRAIN, round_robin=K)]
        cv.append(played[min(K, TRAIN - 1):].mean())
    return play(lams[int(np.argmin(cv))], T, round_robin=TRAIN)


def evaluate(name, alpha, seeds):
    windows = GRIDS[alpha]
    t0 = time.time()
    y, mu, sig, ctx, S, horizon = prepare(name, alpha, windows)
    T = len(y)
    s = np.abs(y - mu) / sig
    z_fb = norm.isf(alpha / 2)
    gammas = [0.0] + list(ACI_FAMILY_GAMMAS)
    experts = [(n, g) for n in windows for g in gammas]
    E = len(experts)
    g_arr = np.array([g for _, g in experts])
    w_of = np.array([windows.index(n) for n, _ in experts])
    levels = np.full(E, alpha)

    exp_lo = np.zeros((T, E)); exp_hi = np.zeros((T, E))
    base = {f"{m}-{n}": np.zeros((T, 2)) for m in ("AgACI", "DtACI") for n in windows}
    agaci = {n: _AgACI(alpha, ACI_FAMILY_GAMMAS) for n in windows}
    dtaci = {n: _DtACI(alpha, ACI_FAMILY_GAMMAS) for n in windows}
    p1 = np.zeros((T, 2)); agg_lo, agg_hi = _MLPoly(E), _MLPoly(E)

    for t in range(T):
        bufs = [np.sort(s[max(0, t - n):t]) for n in windows]
        hw = np.empty(E)
        for k, b in enumerate(bufs):
            sel = w_of == k
            if len(b) >= 2:
                m = len(b)
                idx = np.clip(np.ceil((m + 1) * (1 - levels[sel])).astype(int), 1, m) - 1
                hw[sel] = b[idx] * sig[t]
            else:
                hw[sel] = z_fb * sig[t]
        exp_lo[t], exp_hi[t] = mu[t] - hw, mu[t] + hw
        miss = (y[t] < exp_lo[t]) | (y[t] > exp_hi[t])

        for k, n in enumerate(windows):
            b, fb = bufs[k], z_fb * sig[t]
            base[f"AgACI-{n}"][t] = agaci[n].interval(mu[t], b, sig[t], fb)
            agaci[n].update(y[t])
            lvl = dtaci[n].level()
            h = _conformal_quantile(b, lvl) * sig[t] if len(b) >= 2 else fb
            base[f"DtACI-{n}"][t] = (mu[t] - h, mu[t] + h)
            if len(b) >= 2:
                sc = abs(y[t] - mu[t]) / sig[t]
                errs = np.array([0.0 if sc <= _conformal_quantile(b, a) else 1.0
                                 for a in dtaci[n].levels])
                dtaci[n].update(_covering_level(b, sc), errs)

        p1[t] = (agg_lo.predict(exp_lo[t]), agg_hi.predict(exp_hi[t]))
        agg_lo.update(exp_lo[t], float(y[t] < p1[t, 0]) - alpha / 2)
        agg_hi.update(exp_hi[t], float(y[t] < p1[t, 1]) - (1 - alpha / 2))
        levels += g_arr * (alpha - miss.astype(float))

    is_exp = interval_score(exp_lo, exp_hi, y[:, None], alpha)
    fixed = [experts.index((n, 0.0)) for n in windows]
    assert np.allclose(is_exp[:, fixed], S), "gamma=0 experts must equal the candidate windows"

    lams = polylog_lambda_grid(alpha=alpha, horizon=T)
    methods = {}

    def summarize(lo, hi):
        sc = interval_score(lo, hi, y, alpha)[TRAIN:]
        cov = ((lo <= y) & (y <= hi))[TRAIN:]
        return sc, float(cov.mean()), float((hi - lo)[TRAIN:].mean())

    for key, iv in list(base.items()) + [("P1 AgACI-W", p1)]:
        sc, cov, wid = summarize(iv[:, 0], iv[:, 1])
        methods[key] = {"scores": sc, "coverage": cov, "width": wid}
    for key, cols in [("P2 CTS-W", list(range(E))), ("CTS-5", fixed)]:
        runs = []
        for seed in seeds:
            picks = np.array(cols)[run_selector(is_exp[:, cols], ctx, seed, lams)]
            lo, hi = exp_lo[np.arange(T), picks], exp_hi[np.arange(T), picks]
            runs.append(summarize(lo, hi))
        methods[key] = {"scores": np.mean([r[0] for r in runs], axis=0),
                        "coverage": float(np.mean([r[1] for r in runs])),
                        "width": float(np.mean([r[2] for r in runs]))}

    n_eval = T - TRAIN
    bw = int(max(horizon - 1, np.floor(4 * (n_eval / 100) ** (2 / 9))))
    best = {m: min((k for k in methods if k.startswith(m + "-")),
                   key=lambda k: methods[k]["scores"].mean()) for m in ("AgACI", "DtACI")}
    out = {"dataset": name, "alpha": alpha, "T_eval": n_eval, "bandwidth": bw,
           "seconds": round(time.time() - t0, 1), "best_baseline": best,
           "methods": {k: {"mean_score": float(v["scores"].mean()),
                           "coverage": v["coverage"], "width": v["width"]}
                       for k, v in methods.items()},
           "tests": {}}
    for cand in ("P1 AgACI-W", "P2 CTS-W", "CTS-5"):
        for m, bkey in best.items():
            stat, p = diebold_mariano_test(methods[cand]["scores"], methods[bkey]["scores"],
                                           alternative="two-sided", h=bw + 1)
            out["tests"][f"{cand} vs {bkey}"] = {
                "improvement_pct": 100 * (methods[bkey]["scores"].mean()
                                          - methods[cand]["scores"].mean())
                / methods[bkey]["scores"].mean(),
                "p_two_sided": float(p)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, choices=sorted(GRIDS), required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--output", default="results/prototype_window_level")
    args = ap.parse_args()
    seeds = [42 + i for i in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=len(args.datasets)) as ex:
        results = list(ex.map(evaluate, args.datasets,
                              [args.alpha] * len(args.datasets),
                              [seeds] * len(args.datasets)))
    Path(args.output).mkdir(parents=True, exist_ok=True)
    tag = f"{args.alpha:.2f}".replace(".", "")
    path = Path(args.output) / f"alpha_{tag}.json"
    path.write_text(json.dumps(results, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
