"""Anytime-valid comparison of interval and quantile forecasts.

The score difference d_t of two interval (or quantile) forecasts issued
before y_t lies in a range [a_t, b_t] that is known before y_t is observed,
and it equals a predictable centre c_t (the width difference) unless y_t
falls outside one of the intervals (Lemma 1 of main.tex).  This module
turns those facts into confidence sequences for the running average of
conditional expected differences,

    Delta_bar_t = (1/t) sum_{s<=t} E[d_s | G_s],

with no assumption on the outcome distribution.

  eb_cs        empirical-Bernstein mixture centred at c_t (Theorem 1):
               its width adapts to sum_s (d_s - c_s)^2, which is nonzero
               only on rounds with a miss.
  hoeffding_cs Hoeffding mixture with the same ranges: its width depends on
               the ranges only.
  asymp_cs     Gaussian-mixture asymptotic confidence sequence of
               Waudby-Smith et al. (2024); not finite-sample valid.
  dm_stat      fixed-sample Diebold-Mariano statistic with a Newey-West
               variance; not anytime-valid.
"""
import numpy as np

K_GRID = 20  # lambda_k = 2^{-k} lambda_0, k = 0, ..., K_GRID - 1


def interval_score(lo, hi, y, alpha):
    return (hi - lo) + (2 / alpha) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))


def interval_range(loA, hiA, loB, hiB, alpha):
    """Range [a, b] and centre c of IS(A; y) - IS(B; y) over all y.

    c is the value when y lies in both intervals (the width difference)."""
    c = (hiA - loA) - (hiB - loB)
    m_lo = np.minimum(0, np.minimum(loA - loB, hiB - hiA))
    m_hi = np.maximum(0, np.maximum(loA - loB, hiB - hiA))
    return c + (2 / alpha) * m_lo, c + (2 / alpha) * m_hi, c


def pinball(q, y, tau):
    return np.maximum(tau * (y - q), (tau - 1) * (y - q))


def pinball_range(qA, qB, tau):
    """Range [a, b] and centre c of rho_tau(y - qA) - rho_tau(y - qB).

    c is the value on the side of both quantiles that has probability
    1 - tau (above both for tau < 1/2, below both otherwise)."""
    u, v = tau * (qB - qA), (1 - tau) * (qA - qB)
    c = u if tau < 0.5 else v
    return np.minimum(u, v), np.maximum(u, v), c


def _psi_ratio(u):
    """psi_E(u) / u^2 with psi_E(u) = -log(1 - u) - u; 1/2 at u = 0, inf for u >= 1."""
    u = np.asarray(u, dtype=float)
    out = np.full(u.shape, np.inf)
    small = u < 1e-4
    out[small] = 0.5 + u[small] / 3 + u[small] ** 2 / 4
    mid = (~small) & (u < 1)
    out[mid] = (-np.log1p(-u[mid]) - u[mid]) / u[mid] ** 2
    return out


def lambda_grid(lam0, K=K_GRID):
    return lam0 * 2.0 ** -np.arange(K)


def _lower_bound(x, pen, lam, delta, idx=None, w=None):
    """Largest m with sum_k w_k exp(lam_k (S_t - t m) - V_kt) >= 1/delta.

    pen is K x T (per-round penalties, inf = component discarded); V_kt
    is their cumulative sum; w are fixed weights summing to at most one
    (default 1/K).  Evaluated at the rounds idx (default: all).
    """
    K, T = pen.shape
    logw = np.full(K, -np.log(K)) if w is None else np.log(np.asarray(w, dtype=float))
    idx = np.arange(T) if idx is None else np.asarray(idx)
    V = np.cumsum(pen, axis=1)[:, idx]
    S = np.cumsum(x)[idx]
    t = (idx + 1).astype(float)
    c = logw[:, None] + lam[:, None] * S[None, :] - V
    alive = np.isfinite(c)
    logd = np.log(1 / delta)
    lt = lam[:, None] * t[None, :]
    lo_k = np.where(alive, (c - logd) / lt, -np.inf)
    hi_k = np.where(alive, (c - logd + np.log(K)) / lt, -np.inf)
    lo, hi = lo_k.max(0), hi_k.max(0)
    ok = np.isfinite(lo)
    lo, hi = np.where(ok, lo, 0.0), np.where(ok, hi, 0.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        z = np.where(alive, c - lt * mid[None, :], -np.inf)
        zmax = np.where(ok, z.max(0), 0.0)
        with np.errstate(divide="ignore"):
            logM = zmax + np.log(np.exp(z - zmax).sum(0))
        above = logM >= logd
        lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
    return np.where(ok, lo, -np.inf)


def _eb_penalty(x, a, c, lam):
    """lam^2 psi_E(lam R_t)/(lam R_t)^2 (x_t - c_t)^2, R_t = c_t - a_t >= 0."""
    R = c - a
    D2 = (x - c) ** 2
    u = lam[:, None] * R[None, :]
    with np.errstate(invalid="ignore"):
        pen = lam[:, None] ** 2 * _psi_ratio(u) * D2[None, :]
    pen[np.isnan(pen)] = np.inf                              # inf * 0
    return pen


def eb_cs(d, a, b, c, delta=0.05, lam0=None, idx=None):
    """Two-sided empirical-Bernstein confidence sequence for Delta_bar_t.

    d: score differences; a <= d <= b: predictable bounds; c in [a, b]:
    predictable centre.  lam0 sets the lambda grid and must not depend on
    the outcomes (default: 1 / (2 (b_0 - a_0)), fixed by the first range).
    Returns (lower, upper) at the rounds idx, valid simultaneously over all
    rounds with probability at least 1 - delta.
    """
    d, a, b, c = map(lambda v: np.asarray(v, dtype=float), (d, a, b, c))
    if lam0 is None:
        lam0 = 1 / (2 * max(b[0] - a[0], 1e-12))
    lam = lambda_grid(lam0)
    lower = _lower_bound(d, _eb_penalty(d, a, c, lam), lam, delta / 2, idx)
    upper = -_lower_bound(-d, _eb_penalty(-d, -b, -c, lam), lam, delta / 2, idx)
    return lower, upper


def tuned_grid(v_hat, horizon, r_lo, r_hi, delta=0.05, K=K_GRID, w_star=0.5):
    """Grid and weights for eb_cs tuned to a planned horizon.

    v_hat: per-round mean of (d - c)^2 and r_lo, r_hi: typical ranges below
    and above the centre, all estimated before evaluation (warm-up); horizon:
    planned number of rounds.  The rate-optimal lambda for a fixed horizon is
    sqrt(2 l / (horizon v_hat)) with l = log(2 / (delta w_star)); it receives
    weight w_star, capped so that lambda r <= 1/2, and the remaining weight is
    spread uniformly over lambda* 2^j, j = -K/2, ..., K/2 - 1, j != 0.
    Returns (lam_lower, w_lower, lam_upper, w_upper); the weights must not
    depend on the evaluated outcomes for the sequence to be valid."""
    ell = np.log(2 / (delta * w_star))
    out = []
    for r in (r_lo, r_hi):
        lam_star = np.sqrt(2 * ell / (horizon * max(v_hat, 1e-300)))
        lam_star = min(lam_star, 1 / (2 * max(r, 1e-300)))
        j = np.arange(-(K // 2), K - K // 2)
        lam = lam_star * 2.0 ** j
        w = np.where(j == 0, w_star, (1 - w_star) / (K - 1))
        out += [lam, w]
    return tuple(out)


def eb_cs_tuned(d, a, b, c, grid, delta=0.05, idx=None):
    """eb_cs with the grid and weights of tuned_grid."""
    d, a, b, c = map(lambda v: np.asarray(v, dtype=float), (d, a, b, c))
    lam_l, w_l, lam_u, w_u = grid
    lower = _lower_bound(d, _eb_penalty(d, a, c, lam_l), lam_l, delta / 2, idx, w_l)
    upper = -_lower_bound(-d, _eb_penalty(-d, -b, -c, lam_u), lam_u, delta / 2, idx, w_u)
    return lower, upper


def eb_test_split(d, a, b, c, delta=0.05, frac=0.1, q=0.9):
    """Tuned EB sequence after a tuning split: the first frac of the rounds
    estimate the variance proxy and typical ranges, which fix the grid of
    tuned_grid for a horizon equal to the remaining rounds; the sequence is
    then run on the remaining rounds.  Returns its (lower, upper) bounds at
    the last round; they cover the average of E[d_t | G_t] over the remaining
    rounds."""
    d, a, b, c = map(lambda v: np.asarray(v, dtype=float), (d, a, b, c))
    n0 = max(int(frac * len(d)), 1)
    v_hat = np.mean((d[:n0] - c[:n0]) ** 2)
    grid = tuned_grid(v_hat, len(d) - n0, np.quantile((c - a)[:n0], q),
                      np.quantile((b - c)[:n0], q), delta)
    L, U = eb_cs_tuned(d[n0:], a[n0:], b[n0:], c[n0:], grid, delta, idx=[len(d) - n0 - 1])
    return float(L[0]), float(U[0])


def hoeffding_cs(d, a, b, delta=0.05, lam0=None, idx=None):
    """Two-sided Hoeffding confidence sequence with the same ranges and grid."""
    d, a, b = map(lambda v: np.asarray(v, dtype=float), (d, a, b))
    if lam0 is None:
        lam0 = 1 / (2 * max(b[0] - a[0], 1e-12))
    lam = lambda_grid(lam0, K=2 * K_GRID)   # no range constraint: wider grid
    pen = lam[:, None] ** 2 * ((b - a) ** 2 / 8)[None, :]
    lower = _lower_bound(d, pen, lam, delta / 2, idx)
    upper = -_lower_bound(-d, pen, lam, delta / 2, idx)
    return lower, upper


def asymp_cs(d, delta=0.05, t_opt=500, idx=None):
    """Gaussian-mixture asymptotic CS (Waudby-Smith et al. 2024), tuned for t_opt."""
    d = np.asarray(d, dtype=float)
    t = np.arange(1, len(d) + 1)
    mean = np.cumsum(d) / t
    var = np.maximum(np.cumsum(d ** 2) / t - mean ** 2, 0)
    rho2 = (-2 * np.log(delta) + np.log(-2 * np.log(delta) + 1)) / t_opt
    rad = np.sqrt(var * 2 * (t * rho2 + 1) / (t ** 2 * rho2)
                  * np.log(np.sqrt(t * rho2 + 1) / delta))
    idx = slice(None) if idx is None else idx
    return (mean - rad)[idx], (mean + rad)[idx]


def dm_stat(d, lag=0):
    """Diebold-Mariano t statistic for mean(d) with a Newey-West variance."""
    d = np.asarray(d, dtype=float)
    T = len(d)
    e = d - d.mean()
    v = e @ e / T
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / T
    m = d.mean()
    if v <= 1e-14 * max(1.0, m * m):
        return np.sign(m) * np.inf if m != 0 else 0.0
    return m / np.sqrt(v / T)


def fz0(v, e, y, tau):
    """FZ0 loss of Patton, Ziegel and Chen (2019) for (VaR, ES) at lower level tau,
    with e <= v < 0."""
    return -(1 / (tau * e)) * (y <= v) * (v - y) + v / e + np.log(-e) - 1


def fz0_range(vA, eA, vB, eB, tau):
    """Range [a, b] and centre c of FZ0(A; y) - FZ0(B; y) over all y.

    c is the value when y exceeds both VaRs.  The deviation from c is
    sA (vA - y)_+ - sB (vB - y)_+ with s = 1 / (tau |e|): it is bounded
    below (a finite, b = inf) when sA >= sB, i.e. when A's ES is the less
    conservative one, and bounded above (a = -inf) when sA <= sB."""
    c = (vA / eA + np.log(-eA)) - (vB / eB + np.log(-eB))
    sA, sB = -1 / (tau * eA), -1 / (tau * eB)
    a = np.where(sA >= sB, c - sB * np.maximum(vB - vA, 0), -np.inf)
    b = np.where(sA <= sB, c + sA * np.maximum(vA - vB, 0), np.inf)
    return a, b, c


def fz0_floored(vA, eA, vB, eB, y, tau, ell):
    """FZ0 difference with the outcome floored at ell <= min(vA, vB), and its
    range [a, b] and centre c over y >= ell.

    Flooring lowers the difference on days when A's ES is the less
    conservative (its deviation increases as y decreases) and raises it on
    the other days, so a lower bound on the first kind of day and an upper
    bound on the second remain valid for the unfloored difference."""
    yf = np.maximum(y, ell)
    d = fz0(vA, eA, yf, tau) - fz0(vB, eB, yf, tau)
    pts = [fz0(vA, eA, p, tau) - fz0(vB, eB, p, tau) for p in (ell, vA, vB, np.maximum(vA, vB) + 1)]
    pts = np.stack(pts)
    c = pts[-1]
    return d, pts.min(0), pts.max(0), c
