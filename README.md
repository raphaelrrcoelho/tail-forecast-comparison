# Comparing tail forecasts: code and data

Code, data and results for "Comparing tail forecasts: the number of
exceedances is the sample size" (Raphael Coelho).

- `notes/tail_comparison/`: the paper's scripts, results and LaTeX sources.
  Its `README.md` maps each script to the section, table or figure it
  produces, and gives the order in which to run them.
- `conformal_ts/` and `scripts/`: the online conformal methods used in
  Section 8, called by `notes/tail_comparison/extract_intervals.py`.

Install the requirements (`pip install -r requirements.txt`) and run the
scripts from `notes/tail_comparison/`. The seven forecasting data sets are
downloaded from their public sources on first use. The index prices
(Yahoo Finance, downloaded on 27 October 2026) are included as
`notes/tail_comparison/data/returns.csv`, so the expected-shortfall results
do not depend on later revisions of the Yahoo data.

`results/` holds the outputs behind every number in the paper; `figures.py`
rebuilds the tables and figures from them without rerunning anything.

The code is released under the MIT License (see `LICENSE`).
