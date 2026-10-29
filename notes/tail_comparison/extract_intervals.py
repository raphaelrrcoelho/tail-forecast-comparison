#!/usr/bin/env python3
"""Save the per-round intervals of every online conformal method.

Runs scripts/confirm_window_level.evaluate (matched protocol, seed 42) on each
dataset and level and writes data/{dataset}_a{level}.npz with the outcomes y,
the point forecast mu, the local scale sigma and one (T, 2) array per method.
The first TRAIN rounds are warm-up and are dropped by the analysis scripts.

Usage: python notes/tail_comparison/extract_intervals.py [--alphas 0.1 0.05 0.01]
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import confirm_window_level as cw  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"


def run(name, alpha):
    _, ex = cw.evaluate(name, alpha, [42], keep_intervals=True)
    tag = f"{alpha:.2f}".replace(".", "")
    arrays = {"y": ex["y"], "mu": ex["mu"], "sigma": ex["sigma"],
              "train": np.array(cw.TRAIN), "alpha": np.array(alpha)}
    arrays.update({f"iv::{k}": v for k, v in ex["iv"].items()})
    np.savez_compressed(OUT / f"{name}_a{tag}.npz", **arrays)
    return name, alpha, len(ex["y"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.10, 0.05, 0.01])
    ap.add_argument("--datasets", nargs="+", default=cw.proto.DATASETS)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    jobs = [(d, a) for a in args.alphas for d in args.datasets]
    with ProcessPoolExecutor(max_workers=min(len(jobs), 7)) as ex:
        for name, alpha, T in ex.map(run, *zip(*jobs)):
            print(f"{name} alpha={alpha}: T={T}", flush=True)


if __name__ == "__main__":
    main()
