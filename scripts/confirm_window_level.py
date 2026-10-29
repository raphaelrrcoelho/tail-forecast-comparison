#!/usr/bin/env python3
"""Pre-registered confirmation test for P1 (AgACI-W).

See results/prototype_window_level/PREREGISTRATION.md for the claim, the
protocol and the pass criteria, fixed before these runs.  Every method runs
on the matched protocol: one point forecast mu_t, one local scale sigma_t
and one stream of standardized scores s_t = |y_t - mu_t| / sigma_t; every
interval is mu_t +- sigma_t q_t.

Methods:
  P1 AgACI-W        ML-Poly bound aggregation over ACI experts on every
                    (window, step size) pair (scripts/prototype_window_level.py)
  P1-fixed          ablation: the same aggregation over fixed-level experts
  AgACI-n, DtACI-n  each reference window n; the decision uses the best n
  SAOCP             Bhatnagar et al. (2023), ported from
                    github.com/salesforce/online_conformal (Apache-2.0)
  QT-lr, PI-lr      quantile tracking and PI control, Angelopoulos, Candes &
                    Tibshirani (2023), ported from
                    github.com/aangelopoulos/conformal-time-series (MIT);
                    the decision uses the best learning rate lr
  CTS-5             the CIS-TS selector over fixed-level windows (reference)
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import prototype_window_level as proto  # noqa: E402
from conformal_ts.evaluation.metrics import diebold_mariano_test  # noqa: E402
from conformal_ts.experiments.bandit_experiment import (  # noqa: E402
    ACI_FAMILY_GAMMAS, _AgACI, _DtACI, _MLPoly, _conformal_quantile,
    _covering_level)
from conformal_ts.theory import polylog_lambda_grid  # noqa: E402

GRIDS = {0.20: [25, 50, 100, 200, 500], 0.10: [25, 50, 100, 200, 500],
         0.05: [25, 50, 100, 200, 500], 0.01: [100, 200, 500, 1000, 2000]}
PID_LRS = [1.0, 0.5, 0.1, 0.05]      # the authors' grid, minus lr = 0
PID_BURNIN, PID_CSAT, PID_KI = 100, 5.0, 10.0
TRAIN = proto.TRAIN


def pinball(y, yhat, q):
    return np.maximum(q * (y - yhat), (1 - q) * (yhat - y))


def pinball_grad(y, yhat, q):
    return -q * (y > yhat) + (1 - q) * (y < yhat)


class _SAOCPExpert:
    """Scale-free OGD learner with a geometric lifetime (reference: _OGD)."""

    def __init__(self, t, scale, alpha, yhat0, g=8):
        self.scale, self.base_lr, self.alpha = scale, scale / np.sqrt(3), alpha
        self.yhat, self.grad_norm = yhat0, 0.0
        u = 0
        while t % 2 == 0:
            t //= 2
            u += 1
        self.lifetime = g * 2 ** u
        self.z = self.wz = 0.0
        self.s_t = 0

    @property
    def expired(self):
        return self.s_t > self.lifetime

    @property
    def w(self):
        return 0.0 if self.s_t == 0 else self.z / self.s_t * (1 + self.wz)

    def update(self, y, meta_loss):
        w = self.w
        loss = pinball(y, self.yhat, 1 - self.alpha)
        g = np.clip((meta_loss - loss) / self.scale / max(self.alpha, 1 - self.alpha),
                    -1 * (w > 0), 1)
        self.z += g
        self.wz += g * w
        self.s_t += 1
        grad = pinball_grad(y, self.yhat, 1 - self.alpha)
        self.grad_norm += grad ** 2
        if self.grad_norm != 0:
            self.yhat = max(0.0, self.yhat - self.base_lr / np.sqrt(self.grad_norm) * grad)


class SAOCP:
    """Strongly adaptive online conformal prediction (reference: SAOCP)."""

    def __init__(self, alpha, calibration_scores, lifetime=8):
        self.alpha, self.coverage, self.g = alpha, 1 - alpha, lifetime
        self.t = 1
        self.experts = {}
        self.scale = (float(np.max(np.abs(calibration_scores))) * np.sqrt(3)
                      if len(calibration_scores) else 1.0)
        for s in calibration_scores:
            self.update(s)

    def _p(self):
        prior = {t: 1 / (t ** 2 * (1 + np.floor(np.log2(t)))) for t in self.experts}
        z = sum(prior.values())
        prior = {t: v / z for t, v in prior.items()}
        p = {t: prior[t] * max(0.0, e.w) for t, e in self.experts.items()}
        total = sum(p.values())
        return {t: v / total for t, v in p.items()} if total > 0 else prior

    def predict(self):
        if not self.experts:
            return 0.0
        p = self._p()
        return float(sum(p[t] * e.yhat for t, e in self.experts.items()))

    def update(self, s):
        s_hat = self.predict()
        for t in [k for k, e in self.experts.items() if e.expired]:
            self.experts.pop(t)
        self.experts[self.t] = _SAOCPExpert(self.t, self.scale, self.alpha, s_hat, self.g)
        meta_loss = pinball(s, self.predict(), self.coverage)
        for e in self.experts.values():
            e.update(s, meta_loss)
        self.t += 1


def pid_quantiles(scores, alpha, lr, integrate, q0):
    """Quantile tracking, optionally with the log-saturated integrator
    (reference: quantile_integrator_log_scorecaster, ahead = 1, no
    scorecaster).  q0 warm-starts the tracker at the calibration quantile."""
    T = len(scores)
    q = np.zeros(T)
    qt = q0
    q[0] = qt
    misses = 0.0
    for t in range(T):
        lo = max(t - PID_BURNIN, 0)
        lr_t = lr * (scores[lo:t].max() - scores[lo:t].min()) if t > 0 else lr
        covered = q[t] >= scores[t]
        misses += 0.0 if covered else 1.0
        grad = alpha if covered else -(1 - alpha)
        integ = 0.0
        if integrate:
            x = misses - (t + 1) * alpha
            arg = x * np.log(t + 2) / (PID_CSAT * (t + 2))
            integ = PID_KI * (np.tan(arg) if abs(arg) < np.pi / 2 else np.sign(arg) * np.inf)
        if t < T - 1:
            qt = qt - lr_t * grad
            q[t + 1] = qt + integ
    return q


def evaluate(name, alpha, seeds, with_p2=False, keep_intervals=False):
    """Run every method on one dataset.  With keep_intervals, also return
    the per-round intervals (first seed for the selectors) for analysis."""
    windows = GRIDS[alpha]
    t0 = time.time()
    y, mu, sig, ctx, S, horizon = proto.prepare(name, alpha, windows)
    T = len(y)
    s = np.abs(y - mu) / sig
    z_fb = norm.isf(alpha / 2)
    gammas = [0.0] + list(ACI_FAMILY_GAMMAS)
    experts = [(n, g) for n in windows for g in gammas]
    E = len(experts)
    g_arr = np.array([g for _, g in experts])
    w_of = np.array([windows.index(n) for n, _ in experts])
    fixed = [experts.index((n, 0.0)) for n in windows]
    levels = np.full(E, alpha)

    exp_lo = np.zeros((T, E)); exp_hi = np.zeros((T, E))
    iv = {f"{m}-{n}": np.zeros((T, 2)) for m in ("AgACI", "DtACI") for n in windows}
    agaci = {n: _AgACI(alpha, ACI_FAMILY_GAMMAS) for n in windows}
    dtaci = {n: _DtACI(alpha, ACI_FAMILY_GAMMAS) for n in windows}
    iv["P1 AgACI-W"] = np.zeros((T, 2)); iv["P1-fixed"] = np.zeros((T, 2))
    p1 = (_MLPoly(E), _MLPoly(E)); p1f = (_MLPoly(len(fixed)), _MLPoly(len(fixed)))
    iv["SAOCP"] = np.zeros((T, 2))
    saocp = SAOCP(alpha, s[:TRAIN])

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
            iv[f"AgACI-{n}"][t] = agaci[n].interval(mu[t], b, sig[t], fb)
            agaci[n].update(y[t])
            lvl = dtaci[n].level()
            h = _conformal_quantile(b, lvl) * sig[t] if len(b) >= 2 else fb
            iv[f"DtACI-{n}"][t] = (mu[t] - h, mu[t] + h)
            if len(b) >= 2:
                sc = abs(y[t] - mu[t]) / sig[t]
                errs = np.array([0.0 if sc <= _conformal_quantile(b, a) else 1.0
                                 for a in dtaci[n].levels])
                dtaci[n].update(_covering_level(b, sc), errs)

        for key, (lo_agg, hi_agg), cols in (("P1 AgACI-W", p1, slice(None)),
                                            ("P1-fixed", p1f, fixed)):
            lo_e, hi_e = exp_lo[t, cols], exp_hi[t, cols]
            iv[key][t] = (lo_agg.predict(lo_e), hi_agg.predict(hi_e))
            lo_agg.update(lo_e, float(y[t] < iv[key][t, 0]) - alpha / 2)
            hi_agg.update(hi_e, float(y[t] < iv[key][t, 1]) - (1 - alpha / 2))
        levels += g_arr * (alpha - miss.astype(float))

        if t >= TRAIN:  # SAOCP was pre-fed the calibration scores s[:TRAIN]
            d = saocp.predict()
            iv["SAOCP"][t] = (mu[t] - d * sig[t], mu[t] + d * sig[t])
            saocp.update(s[t])

    q0 = float(np.quantile(s[:TRAIN], 1 - alpha))
    for lr in PID_LRS:
        for tag, integ in (("QT", False), ("PI", True)):
            q = pid_quantiles(s, alpha, lr, integ, q0)
            q = np.where(np.isfinite(q), q, np.nan)
            q = np.nan_to_num(q, nan=np.nanmax(s[:TRAIN]) * 10)
            iv[f"{tag}-{lr:g}"] = np.column_stack([mu - q * sig, mu + q * sig])

    is_exp = proto.interval_score(exp_lo, exp_hi, y[:, None], alpha)
    assert np.allclose(is_exp[:, fixed], S), "fixed experts must equal the candidates"

    def summarize(lo, hi):
        sc = proto.interval_score(lo, hi, y, alpha)[TRAIN:]
        cov = ((lo <= y) & (y <= hi))[TRAIN:]
        return sc, float(cov.mean()), float((hi - lo)[TRAIN:].mean())

    methods = {}
    for key, v in iv.items():
        sc, cov, wid = summarize(v[:, 0], v[:, 1])
        methods[key] = {"scores": sc, "coverage": cov, "width": wid}
    lams = polylog_lambda_grid(alpha=alpha, horizon=T)
    runs = []
    for seed in seeds:
        picks = np.array(fixed)[proto.run_selector(is_exp[:, fixed], ctx, seed, lams)]
        runs.append(summarize(exp_lo[np.arange(T), picks], exp_hi[np.arange(T), picks]))
        if keep_intervals and seed == seeds[0]:
            iv["CTS-5"] = np.column_stack([exp_lo[np.arange(T), picks],
                                           exp_hi[np.arange(T), picks]])
            picked = {"CTS-5": [experts[j] for j in picks]}
    methods["CTS-5"] = {"scores": np.mean([r[0] for r in runs], axis=0),
                        "coverage": float(np.mean([r[1] for r in runs])),
                        "width": float(np.mean([r[2] for r in runs]))}
    if with_p2:
        runs = []
        for seed in seeds:
            picks = proto.run_selector(is_exp, ctx, seed, lams)
            runs.append(summarize(exp_lo[np.arange(T), picks], exp_hi[np.arange(T), picks]))
            if keep_intervals and seed == seeds[0]:
                iv["P2 CTS-W"] = np.column_stack([exp_lo[np.arange(T), picks],
                                                  exp_hi[np.arange(T), picks]])
                picked["P2 CTS-W"] = [experts[j] for j in picks]
        methods["P2 CTS-W"] = {"scores": np.mean([r[0] for r in runs], axis=0),
                               "coverage": float(np.mean([r[1] for r in runs])),
                               "width": float(np.mean([r[2] for r in runs]))}

    n_eval = T - TRAIN
    bw = int(max(horizon - 1, np.floor(4 * (n_eval / 100) ** (2 / 9))))

    def best_of(prefix):
        return min((k for k in methods if k.startswith(prefix)),
                   key=lambda k: methods[k]["scores"].mean())

    baselines = {"AgACI": best_of("AgACI-"), "DtACI": best_of("DtACI-"),
                 "SAOCP": "SAOCP", "QT": best_of("QT-"), "PI": best_of("PI-")}
    out = {"dataset": name, "alpha": alpha, "T_eval": n_eval, "bandwidth": bw,
           "seconds": round(time.time() - t0, 1), "baselines": baselines,
           "methods": {k: {"mean_score": float(v["scores"].mean()),
                           "coverage": v["coverage"], "width": v["width"]}
                       for k, v in methods.items()},
           "tests": {}}
    cands = ["P1 AgACI-W", "P1-fixed", "CTS-5"] + (["P2 CTS-W"] if with_p2 else [])
    for cand in cands:
        for fam, bkey in baselines.items():
            _, p = diebold_mariano_test(methods[cand]["scores"], methods[bkey]["scores"],
                                        alternative="two-sided", h=bw + 1)
            b_mean = methods[bkey]["scores"].mean()
            out["tests"][f"{cand} vs {fam}"] = {
                "baseline": bkey,
                "improvement_pct": 100 * (b_mean - methods[cand]["scores"].mean()) / b_mean,
                "p_two_sided": float(p)}
    if keep_intervals:
        return out, {"iv": iv, "y": y, "mu": mu, "sigma": sig, "picked": picked,
                     "p1_weights": None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, choices=sorted(GRIDS), required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--datasets", nargs="+", default=proto.DATASETS)
    ap.add_argument("--output", default="results/prototype_window_level/confirm")
    ap.add_argument("--with-p2", action="store_true",
                    help="also run P2 (not part of the pre-registered test)")
    args = ap.parse_args()
    seeds = [42 + i for i in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=len(args.datasets)) as ex:
        results = list(ex.map(evaluate, args.datasets,
                              [args.alpha] * len(args.datasets),
                              [seeds] * len(args.datasets),
                              [args.with_p2] * len(args.datasets)))
    Path(args.output).mkdir(parents=True, exist_ok=True)
    tag = f"{args.alpha:.2f}".replace(".", "")
    path = Path(args.output) / f"alpha_{tag}.json"
    path.write_text(json.dumps(results, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
