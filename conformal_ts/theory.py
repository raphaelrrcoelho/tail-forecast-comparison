"""
Theory-derived hyperparameters for Conformal Thompson Sampling.

The ridge prior precision lambda is set per Theorem 1 of the regret-bound
analysis (docs/regret_bound.tex): any member of the polylog family
lambda = (log T)^q * M / (S * alpha), q >= 0, yields the tight
Bayesian regret rate up to polylog factors. Within this family the
point q = 1 is uniquely bound-optimal, but empirically smaller q often
wins in finite T by trading a slightly looser Freedman correction for
less over-regularization of the posterior.

Rather than committing to a single q a priori, the algorithm selects
among {q=0, q=1/2, q=1} -- i.e. {1/alpha, sqrt(log T)/alpha, log T/alpha}
-- via a warm-up CV pass on the first portion of each stream.  The grid
is exposed here as `polylog_lambda_grid`; `compute_lambda_star` is kept
as a convenience returning the q=1 bound-optimal value.

M and S default to 1 because (i) interval-score rewards are scaled to
unit magnitude and (ii) features are normalized.
"""

from __future__ import annotations

import math
from typing import List


def compute_lambda_star(
    alpha: float,
    horizon: int,
    *,
    outcome_bound: float = 1.0,
    param_norm_bound: float = 1.0,
    c: float = 1.0,
) -> float:
    """Bound-optimal (q=1) ridge prior precision per Theorem 1.

    Args:
        alpha: Miscoverage level (1 - target coverage), in (0, 1).
        horizon: Total bandit rounds T.
        outcome_bound: M in the bound; default 1.0.
        param_norm_bound: S in the bound; default 1.0.
        c: Absolute constant in the Theta(.) prescription; default 1.0.

    Returns:
        lambda > 0.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if horizon < 2:
        raise ValueError(f"horizon must be >= 2; got {horizon}")
    return c * outcome_bound * math.log(horizon) / (param_norm_bound * alpha)


def polylog_lambda_grid(
    alpha: float,
    horizon: int,
    *,
    outcome_bound: float = 1.0,
    param_norm_bound: float = 1.0,
) -> List[float]:
    """Polylog-family candidates for warm-up CV selection.

    Returns lambdas for q in {0, 1/2, 1}, all of which preserve the
    tilde{O}(d * sqrt(K T / alpha)) Bayesian-regret rate.  The warm-up
    CV picks the empirical winner on a held-in prefix of the stream.

    Args:
        alpha: Miscoverage level (1 - target coverage), in (0, 1).
        horizon: Total bandit rounds T.
        outcome_bound: M in the bound; default 1.0.
        param_norm_bound: S in the bound; default 1.0.

    Returns:
        Three candidate lambdas [1/alpha, sqrt(log T)/alpha, log T/alpha],
        each scaled by M / S.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if horizon < 2:
        raise ValueError(f"horizon must be >= 2; got {horizon}")
    scale = outcome_bound / (param_norm_bound * alpha)
    log_t = math.log(horizon)
    return [
        scale,                      # q = 0
        scale * math.sqrt(log_t),   # q = 1/2
        scale * log_t,              # q = 1
    ]
