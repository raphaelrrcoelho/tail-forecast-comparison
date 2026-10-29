# Comparing tail forecasts: code and data

Reproduces every number, table and figure in `main.tex`.

Requirements: Python 3.11+, numpy, scipy, pandas, matplotlib; `arch` and
`yfinance` for the expected-shortfall study. With uv:

    uv run --no-project --with numpy --with scipy --with pandas \
        --with matplotlib --with arch --with yfinance python <script>

| Script | Paper | Output |
|---|---|---|
| `cs.py` | Sections 3-6 | library: scores, predictable ranges, EB/Hoeffding/asymptotic confidence sequences, DM statistic, FZ0 loss |
| `sim.py` | Section 7 | `results/sim_null.json`, `results/sim_detect.json` |
| `extract_intervals.py` | Section 8 | `data/{dataset}_a{level}.npz` (per-round intervals of the online conformal methods) |
| `empirical.py` | Section 8 | `results/empirical.json` |
| `conditions.py` | Section 8 | `results/conditions.json` (miss rates, rolling miss rates, effect sizes; also a hindsight-oracle diagnostic not reported in the paper) |
| `es_forecasts.py` | Section 9 | `data/returns.csv`, `data/es_{index}.npz` (VaR and ES forecasts) |
| `es_analysis.py [--kappa k]` | Section 9 | `results/es.json`, `results/es_kappa{k}.json` (k = 1.5, 3) |
| `sim_dm.py` | Section 7, Appendix B | `results/sim_dm.json` (HLN, fixed-smoothing and bootstrap-t versions of the DM test; bootstrap power) |
| `es_uncertainty.py` | Section 9 | `results/es_uncertainty.json` (calendar-year cluster bootstrap of the window shares) |
| `es_boot.py` | Section 9 | `results/es_boot.json` (block bootstrap-t version of the window DM tests) |
| `figures.py` | all | `fig_detection.pdf`, `fig_dm.pdf`, `tab_null.tex`, `tab_dmcorr.tex`, `tab_power.tex`, `tab_empirical.tex`, `tab_es.tex` |

Order: `sim.py`, then `sim_dm.py`; `extract_intervals.py`, `empirical.py`, then `conditions.py`;
`es_forecasts.py`, then `es_analysis.py` (three values of kappa), `es_uncertainty.py` and
`es_boot.py`; finally `figures.py`, and
`pdflatex main; bibtex main; pdflatex main; pdflatex main`.

`extract_intervals.py` runs the online conformal methods through
`scripts/confirm_window_level.py` of the parent repository, which also loads
the seven public data sets (sources in Section 8 of the paper). The index
prices are daily closes from Yahoo Finance, January 2000 to September 2026,
downloaded on 27 October 2026; `es_forecasts.py` downloads them once into
`data/returns.csv`, which is included so that the results do not depend on
later revisions of the Yahoo data.
