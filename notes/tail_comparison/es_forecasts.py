#!/usr/bin/env python3
"""One-day-ahead (VaR, ES) forecasts for daily index returns.

Downloads daily closes (Yahoo Finance) once into data/returns.csv, computes
percentage log returns, and writes data/es_{index}.npz with the returns and,
for each model, VaR and ES forecasts at the lower levels in TAUS.  Every
forecast for day t uses returns up to day t-1 only.

Models
  HS-250, HS-500  historical simulation on the last 250 / 500 returns
  EWMA            RiskMetrics variance (lambda = 0.94), normal quantiles
  GARCH-N         GARCH(1,1), normal innovations
  GARCH-t         GARCH(1,1), standardized Student-t innovations
  FHS             filtered historical simulation: GARCH-N volatility times
                  the empirical quantile and tail mean of the standardized
                  residuals in the estimation window
GARCH models use zero mean, are re-estimated every REFIT days on the last
WINDOW returns, and update the variance daily between re-estimations.

Usage: python es_forecasts.py
"""
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
TICKERS = {"S&P 500": "^GSPC", "FTSE 100": "^FTSE", "DAX": "^GDAXI", "Nikkei 225": "^N225",
           "Euro Stoxx 50": "^STOXX50E", "Hang Seng": "^HSI", "Bovespa": "^BVSP"}
START, END = "2000-01-01", "2026-10-01"
TAUS = [0.025, 0.01]
WINDOW, REFIT, LAM = 1000, 20, 0.94
MODELS = ["HS-250", "HS-500", "EWMA", "GARCH-N", "GARCH-t", "FHS"]


def download():
    path = DATA / "returns.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import yfinance as yf
    cols = {}
    for name, tic in TICKERS.items():
        px = yf.download(tic, start=START, end=END, progress=False, auto_adjust=True)["Close"]
        px = px.squeeze().dropna()
        cols[name] = 100 * np.log(px).diff()
    df = pd.DataFrame(cols)
    df.to_csv(path)
    return df


def hs(r, t, n, tau):
    x = np.sort(r[t - n:t])
    k = int(np.ceil(tau * n))
    return x[k - 1], x[:k].mean()


def normal_ve(sig, tau):
    z = stats.norm.ppf(tau)
    return sig * z, -sig * stats.norm.pdf(z) / tau


def t_ve(sig, tau, nu):
    s = np.sqrt((nu - 2) / nu)
    q = stats.t.ppf(tau, nu)
    es = -(stats.t.pdf(q, nu) / tau) * (nu + q ** 2) / (nu - 1)
    return sig * s * q, sig * s * es


def forecast(name):
    from arch import arch_model
    r = download()[name].dropna().to_numpy()
    T = len(r)
    out = {m: {tau: np.full((T, 2), np.nan) for tau in TAUS} for m in MODELS}
    ew = np.var(r[:250])
    sig2 = {"GARCH-N": None, "GARCH-t": None}
    par, resid_q = {}, {}
    for t in range(WINDOW, T):
        if (t - WINDOW) % REFIT == 0:
            for m, dist in (("GARCH-N", "normal"), ("GARCH-t", "t")):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = arch_model(r[t - WINDOW:t], mean="Zero", vol="GARCH", p=1, q=1,
                                     dist=dist, rescale=False).fit(disp="off")
                p = res.params
                par[m] = (p["omega"], p["alpha[1]"], p["beta[1]"], p.get("nu", np.nan))
                cv = np.asarray(res.conditional_volatility)
                sig2[m] = cv[-1] ** 2              # variance for day t-1
                if m == "GARCH-N":
                    z = np.sort(r[t - WINDOW:t] / cv)
                    resid_q = {tau: (z[int(np.ceil(tau * WINDOW)) - 1],
                                     z[:int(np.ceil(tau * WINDOW))].mean()) for tau in TAUS}
        for m in ("GARCH-N", "GARCH-t"):           # roll the variance to day t
            om, al, be, _ = par[m]
            sig2[m] = om + al * r[t - 1] ** 2 + be * sig2[m]
        ew = LAM * ew + (1 - LAM) * r[t - 1] ** 2 if t > WINDOW else np.var(r[t - 250:t])
        for tau in TAUS:
            out["HS-250"][tau][t] = hs(r, t, 250, tau)
            out["HS-500"][tau][t] = hs(r, t, 500, tau)
            out["EWMA"][tau][t] = normal_ve(np.sqrt(ew), tau)
            out["GARCH-N"][tau][t] = normal_ve(np.sqrt(sig2["GARCH-N"]), tau)
            out["GARCH-t"][tau][t] = t_ve(np.sqrt(sig2["GARCH-t"]), tau, par["GARCH-t"][3])
            q, e = resid_q[tau]
            out["FHS"][tau][t] = (np.sqrt(sig2["GARCH-N"]) * q, np.sqrt(sig2["GARCH-N"]) * e)
    arrays = {"r": r, "start": np.array(WINDOW)}
    for m in MODELS:
        for tau in TAUS:
            v, e = out[m][tau][:, 0], out[m][tau][:, 1]
            v = np.minimum(v, -1e-8)                 # FZ0 needs e <= v < 0
            arrays[f"{m}::{tau}"] = np.column_stack([v, np.minimum(e, v)])
    np.savez_compressed(DATA / f"es_{name.replace(' ', '_').replace('&', '')}.npz", **arrays)
    return name, T - WINDOW


def main():
    DATA.mkdir(exist_ok=True)
    download()
    with ProcessPoolExecutor(max_workers=len(TICKERS)) as ex:
        for name, n in ex.map(forecast, TICKERS):
            print(f"{name}: {n} forecast days", flush=True)


if __name__ == "__main__":
    main()
