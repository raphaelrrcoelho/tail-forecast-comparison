#!/usr/bin/env python3
"""Bootstrap-t version of the window comparisons in main.tex Section 9.

In each non-overlapping 250-day window, the Diebold-Mariano statistic of
es_analysis.py is compared with the quantiles of its circular block bootstrap
distribution (blocks of 10 days, 999 resamples), with the statistic and its
replicates both studentized by the variance of the block means, instead of
the Newey-West statistic and normal quantiles.  Reports, as es_analysis.py does
for the DM test, the share of significant window-pair tests, the share of those
that favour the less conservative model, and the wrong-sign rates judged
against all other days.

Usage: python es_boot.py
"""
import itertools
import json
from pathlib import Path

import numpy as np

import cs
from es_analysis import WARM, WIN
from es_forecasts import MODELS, TAUS, TICKERS

HERE = Path(__file__).resolve().parent
BLOCK, B_BOOT = 10, 999


def boot_sign(d, rng):
    """-1, 0 or +1: equal-tailed 5% block bootstrap-t test of E d = 0."""
    T = len(d)
    nb = T // BLOCK
    b0 = d[:nb * BLOCK].reshape(nb, BLOCK).mean(axis=1)
    s0 = b0.std(ddof=1) / np.sqrt(nb)
    if s0 == 0:
        return int(np.sign(d.mean()))
    t0 = d.mean() / s0
    starts = rng.integers(0, T, size=(B_BOOT, nb))
    idx = (starts[:, :, None] + np.arange(BLOCK)) % T          # circular blocks
    bm = d[idx].mean(axis=2)                                     # block means
    m = bm.mean(axis=1)
    se = bm.std(axis=1, ddof=1) / np.sqrt(nb)
    tb = np.where(se > 0, (m - d.mean()) / np.where(se > 0, se, 1), 0.0)
    lo, hi = np.quantile(tb, [0.025, 0.975])
    return -1 if t0 < lo else (1 if t0 > hi else 0)


def main():
    rng = np.random.default_rng(11)
    out = {}
    for tau in TAUS:
        rows = []
        for name in TICKERS:
            z = np.load(HERE / "data" / f"es_{name.replace(' ', '_').replace('&', '')}.npz")
            r, s0 = z["r"], int(z["start"])
            fc = {m: z[f"{m}::{tau}"][s0:] for m in MODELS}
            y = r[s0:]
            for A, B in itertools.combinations(MODELS, 2):
                vA, eA = fc[A][:, 0], fc[A][:, 1]
                vB, eB = fc[B][:, 0], fc[B][:, 1]
                d = (cs.fz0(vA, eA, y, tau) - cs.fz0(vB, eB, y, tau))[WARM:]
                eA_, eB_ = eA[WARM:], eB[WARM:]
                for w0 in range(0, len(d) - WIN + 1, WIN):
                    sl = slice(w0, w0 + WIN)
                    sig = boot_sign(d[sl], rng)
                    if sig == 0:
                        rows.append((0, False, False))
                        continue
                    rest = np.delete(d, np.arange(w0, w0 + WIN)).mean()
                    less = 1 if eA_[sl].mean() > eB_[sl].mean() else -1
                    rows.append((sig, sig == -less, sig != np.sign(rest)))
        sig = np.array([x[0] != 0 for x in rows])
        fl = np.array([x[1] for x in rows])[sig]
        wr = np.array([x[2] for x in rows])[sig]
        out[str(tau)] = {"windows": len(rows), "sig": float(sig.mean()),
                         "sig_favours_less": float(fl.mean()),
                         "wrong_less": float(wr[fl].mean()), "wrong_more": float(wr[~fl].mean()),
                         "n_less": int(fl.sum()), "n_more": int((~fl).sum())}
        print(tau, {k: round(v, 3) if isinstance(v, float) else v for k, v in out[str(tau)].items()})
    (HERE / "results" / "es_boot.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
