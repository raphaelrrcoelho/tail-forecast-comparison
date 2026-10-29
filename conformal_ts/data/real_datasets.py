"""
Real Public Dataset Loaders for Bandit Experiments.

Downloads, caches, and prepares publicly available time series datasets
for use with Conformal Thompson Sampling experiments.  Each dataset is
converted into a (contexts, scores_matrix) representation where:

- contexts:       (T, D) feature matrix  -- bandit context at each step
- scores_matrix:  (T, K) interval scores -- one per specification (lookback window)

The specification space consists of different lookback windows for a simple
rolling-statistics forecaster.  At each timestep *t* and for each spec *k*
(lookback window w_k):

    1. Compute rolling_mean and rolling_std over the last w_k observations.
    2. Form a prediction interval [mean - z * std, mean + z * std].
    3. Evaluate interval_score(lower, upper, y_true, alpha).

This produces genuinely different performance across windows depending on
the local regime (trend, volatility, seasonality).

Datasets
--------
1. **ETTh1** -- Electricity Transformer Temperature (hourly), 7 features
   URL: https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv

2. **ETTh2** -- Second transformer station (hourly), 7 features
   URL: https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv

3. **Electricity** -- UCI 370-client electricity consumption (15-min)
   URL: https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip

4. **AustralianElecDemand** -- Synthetic Australian-style electricity demand
   (deterministic seasonal + stochastic; no external download needed)

5. **Weather** -- 21 meteorological indicators (10-min, resampled to hourly)
   URL: https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv
   (We reuse ETTm1 at minute granularity as a weather-proxy dataset.)

References
----------
Zhou et al. (2021) "Informer: Beyond Efficient Transformer for Long
Sequence Time-Series Forecasting", AAAI.
"""

import gzip
import io
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from conformal_ts.evaluation.metrics import interval_score

logger = logging.getLogger(__name__)

__all__ = [
    "RealDatasetConfig",
    "RealDatasetLoader",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RealDatasetConfig:
    """Metadata and download information for a single real dataset."""

    name: str
    url: str
    target_col: str
    freq: str  # pandas frequency alias: 'H', '15min', 'D', etc.
    description: str = ""
    date_col: str = "date"
    separator: str = ","
    decimal: str = "."
    header: int = 0
    # Lookback windows (in native time-steps) used as the specification space.
    # Defaults are overridden per-dataset at registration time.
    lookback_windows: List[int] = field(default_factory=lambda: [24, 48, 96, 168, 336])
    # Number of clients to sample for very large multi-client datasets.
    sample_clients: Optional[int] = None
    # Whether the URL points to a zip file.
    is_zip: bool = False
    # Filename inside the zip (if applicable).
    zip_member: Optional[str] = None
    # Whether the URL points to a gzip file.
    is_gzip: bool = False
    # If True, file has no header row (headerless CSV/TSV).
    no_header: bool = False
    # Seasonal period (in native timesteps) for the seasonal naive forecaster.
    seasonal_period: Optional[int] = None


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, RealDatasetConfig] = {
    "ETTh1": RealDatasetConfig(
        name="ETTh1",
        url="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
        target_col="OT",
        freq="H",
        description="Electricity Transformer Temperature -- station 1, hourly.",
        lookback_windows=[24, 48, 96, 168, 336],
        seasonal_period=24,
    ),
    "ETTh2": RealDatasetConfig(
        name="ETTh2",
        url="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
        target_col="OT",
        freq="H",
        description="Electricity Transformer Temperature -- station 2, hourly.",
        lookback_windows=[24, 48, 96, 168, 336],
        seasonal_period=24,
    ),
    "Electricity": RealDatasetConfig(
        name="Electricity",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip",
        target_col="MT_001",  # Will be replaced by sampled client columns
        freq="15min",
        description="UCI 370-client electricity consumption, 15-min resolution.",
        separator=";",
        decimal=",",
        header=0,
        lookback_windows=[96, 192, 384, 672, 1344],  # Scaled for 15-min
        sample_clients=5,
        is_zip=True,
        zip_member="LD2011_2014.txt",
        seasonal_period=96,
    ),
    "AustralianElecDemand": RealDatasetConfig(
        name="AustralianElecDemand",
        url="",  # Synthesised locally -- no download needed
        target_col="demand",
        freq="30min",
        description="Synthetic Australian-style half-hourly electricity demand.",
        lookback_windows=[48, 96, 192, 336, 672],  # Scaled for 30-min
        seasonal_period=48,
    ),
    "ETTm1": RealDatasetConfig(
        name="ETTm1",
        url="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
        target_col="OT",
        freq="15min",
        description="Electricity Transformer Temperature -- station 1, 15-min.",
        lookback_windows=[96, 192, 384, 672, 1344],
        seasonal_period=96,
    ),
    "ETTm2": RealDatasetConfig(
        name="ETTm2",
        url="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
        target_col="OT",
        freq="15min",
        description="Electricity Transformer Temperature -- station 2, 15-min.",
        lookback_windows=[96, 192, 384, 672, 1344],
        seasonal_period=96,
    ),
    "ExchangeRate": RealDatasetConfig(
        name="ExchangeRate",
        url="https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/exchange_rate/exchange_rate.txt.gz",
        target_col="col_0",
        freq="D",
        description="Daily exchange rates of 8 countries, 1990-2016.",
        lookback_windows=[7, 14, 30, 60, 120],
        is_gzip=True,
        no_header=True,
        seasonal_period=7,
    ),
    "Traffic": RealDatasetConfig(
        name="Traffic",
        url="https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/traffic/traffic.txt.gz",
        target_col="col_0",
        freq="H",
        description="California freeway occupancy rates, 862 sensors, hourly.",
        lookback_windows=[24, 48, 96, 168, 336],
        is_gzip=True,
        no_header=True,
        seasonal_period=24,
    ),
    # --- Adaptive-conformal benchmark datasets -----------------------------
    "EPEX_SPOT": RealDatasetConfig(
        name="EPEX_SPOT",
        url="https://data.open-power-system-data.org/time_series/latest/time_series_60min_singleindex.csv",
        target_col="DE_LU_price_day_ahead",
        freq="H",
        description=(
            "EPEX SPOT day-ahead electricity prices (DE/LU bidding zone), hourly, "
            "via OPSD. Stand-in for AgACI's French EPEX benchmark -- OPSD's latest "
            "snapshot no longer carries FR_price_day_ahead, and DE/LU is the largest "
            "liquid EPEX SPOT zone adjacent to France."
        ),
        lookback_windows=[24, 48, 96, 168, 336],
        seasonal_period=24,
    ),
    "M4_Hourly": RealDatasetConfig(
        name="M4_Hourly",
        url="https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/Train/Hourly-train.csv",
        target_col="series",
        freq="H",
        description="M4 competition hourly track (FACI's headline); first 20 series concatenated.",
        lookback_windows=[24, 48, 96, 168, 336],
        seasonal_period=24,
    ),
    "gefcom_wind": RealDatasetConfig(
        name="gefcom_wind",
        url="",  # Real track file if present, else synthesised locally.
        target_col="power",
        freq="H",
        description="GEFCom2014 wind track, zone 1 (real data when available).",
        lookback_windows=[24, 48, 96, 168, 336],
        seasonal_period=24,
    ),
    "SynthVarBreak": RealDatasetConfig(
        name="SynthVarBreak",
        url="",  # Synthesised locally (see _generate_varbreak).
        target_col="y",
        freq="H",
        description=("Piecewise noise-scale synthetic: level + AR(1) with "
                     "abrupt volatility regime breaks, so the optimal CQR "
                     "calibration window provably shifts across segments. "
                     "seasonal_period=1 gives one-step-ahead evaluation, so "
                     "segment lengths are in evaluation-step units and the "
                     "calibration windows {25..500} are commensurate with "
                     "the 250-450-step regimes."),
        lookback_windows=[24, 48, 96, 168, 336],
        seasonal_period=1,
    ),
    "gefcom_solar": RealDatasetConfig(
        name="gefcom_solar",
        url="",  # Synthesised locally (see _generate_gefcom).
        target_col="power",
        freq="H",
        description="GEFCom2014 solar-track-style hourly power (synthetic fallback).",
        lookback_windows=[24, 48, 96, 168, 336],
        seasonal_period=24,
    ),
    "NN5_Daily": RealDatasetConfig(
        name="NN5_Daily",
        url="https://zenodo.org/records/4656117/files/nn5_daily_dataset_without_missing_values.zip",
        target_col="series",
        freq="D",
        description="Monash archive NN5 daily cash-withdrawal series; first 20 series concatenated.",
        lookback_windows=[7, 14, 30, 60, 120],
        is_zip=True,
        zip_member="nn5_daily_dataset_without_missing_values.tsf",
        seasonal_period=7,
    ),
}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class RealDatasetLoader:
    """Downloads, caches, and prepares real datasets for bandit experiments."""

    REGISTRY: Dict[str, RealDatasetConfig] = _REGISTRY

    def __init__(self, cache_dir: str = "./cache/datasets"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def list_available() -> List[str]:
        """Return the names of all registered datasets."""
        return list(_REGISTRY.keys())

    def load(self, name: str) -> Optional[pd.DataFrame]:
        """
        Load a dataset by name, downloading and caching if needed.

        Parameters
        ----------
        name : str
            One of the keys in ``REGISTRY``.

        Returns
        -------
        pd.DataFrame or None
            DataFrame indexed by datetime with numeric columns.
            Returns ``None`` if the download fails.
        """
        if name not in self.REGISTRY:
            raise ValueError(
                f"Unknown dataset '{name}'. "
                f"Available: {self.list_available()}"
            )

        config = self.REGISTRY[name]

        # Special case: Australian demand is generated, not downloaded.
        if name == "AustralianElecDemand":
            return self._generate_australian_demand()

        # Special case: variance-break synthetic (generated, not downloaded).
        if name == "SynthVarBreak":
            return self._generate_varbreak()

        # Special case: GEFCom2014 (hourly wind/solar).  Prefer the REAL
        # competition data when the raw track files are present under
        # ./data/gefcom2014/ (wind.csv / solar.csv, long format with zone
        # one-hots, as distributed with Hong et al. 2016); otherwise fall
        # back to the synthetic GEFCom-style generator and say so loudly.
        import os
        force_synth = os.environ.get("CIS_GEFCOM_FORCE_SYNTHETIC", "") == "1"
        if name in ("gefcom_wind", "gefcom_solar"):
            track = "wind" if name == "gefcom_wind" else "solar"
            real = None if force_synth else self._load_gefcom_real(track=track)
            if real is not None:
                logger.info("%s: using REAL GEFCom2014 %s track (zone 1)",
                            name, track)
                return real
            logger.warning("%s: SYNTHETIC generator (forced=%s)",
                           name, force_synth)
            return self._generate_gefcom(track=track)

        cache_path = self.cache_dir / f"{name}.csv"
        if cache_path.exists():
            logger.info("Loading cached %s from %s", name, cache_path)
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return df

        logger.info("Downloading %s from %s ...", name, config.url)
        try:
            df = self._download(config)
        except Exception as exc:
            logger.error("Failed to download %s: %s", name, exc)
            return None

        if df is not None:
            df.to_csv(cache_path)
            logger.info("Cached %s to %s", name, cache_path)

        return df

    def prepare_bandit_experiment(
        self,
        name: str,
        num_specs: Optional[int] = None,
        alpha: float = 0.10,
        z: float = 1.96,
        warmup_fraction: float = 0.05,
    ) -> Optional[Dict]:
        """
        Prepare a dataset for the bandit specification-selection experiment.

        Parameters
        ----------
        name : str
            Dataset name (key in REGISTRY).
        num_specs : int, optional
            If given, use only the first ``num_specs`` lookback windows.
        alpha : float
            Miscoverage rate for the interval score (default 0.10 = 90 %).
        z : float
            Number of standard deviations for the prediction interval
            (default 1.96 ~ 97.5 % two-sided).
        warmup_fraction : float
            Fraction of timesteps to skip at the start (before the longest
            lookback window has enough data).

        Returns
        -------
        dict or None
            ``contexts``      -- np.ndarray, shape (T, D)
            ``scores_matrix`` -- np.ndarray, shape (T, K)
            ``lookback_windows`` -- list[int]
            ``timestamps``    -- pd.DatetimeIndex
            ``target_values`` -- np.ndarray, shape (T,)
            ``dataset_name``  -- str
            ``freq``          -- str
            Returns ``None`` when the dataset cannot be loaded.
        """
        df = self.load(name)
        if df is None:
            return None

        config = self.REGISTRY[name]
        target_col = config.target_col

        # For Electricity we average sampled client columns into a target.
        if name == "Electricity" and target_col not in df.columns:
            target_col = df.columns[0]

        if target_col not in df.columns:
            logger.error(
                "Target column '%s' not found in %s. Columns: %s",
                target_col, name, list(df.columns),
            )
            return None

        target = df[target_col].values.astype(np.float64)

        lookback_windows = list(config.lookback_windows)
        if num_specs is not None:
            lookback_windows = lookback_windows[:num_specs]

        max_lb = max(lookback_windows)
        warmup = max(int(len(target) * warmup_fraction), max_lb + 1)

        T_total = len(target) - warmup
        if T_total <= 0:
            logger.error("Dataset %s is too short for the requested lookback windows.", name)
            return None

        # --- Build scores_matrix (T, K) -----------------------------------
        K = len(lookback_windows)
        scores_matrix = np.full((T_total, K), np.nan)

        for k, w in enumerate(lookback_windows):
            for t_idx in range(T_total):
                t = warmup + t_idx
                history = target[t - w : t]
                if len(history) < 2:
                    continue
                mu = np.mean(history)
                sigma = np.std(history, ddof=1)
                sigma = max(sigma, 1e-8)
                lower = mu - z * sigma
                upper = mu + z * sigma
                y_true = target[t]
                scores_matrix[t_idx, k] = interval_score(
                    np.array([lower]),
                    np.array([upper]),
                    np.array([y_true]),
                    alpha=alpha,
                )[0]

        # --- Build context features (T, D) --------------------------------
        contexts = self.build_context_features(
            df, target, warmup, T_total, config,
        )

        timestamps = df.index[warmup : warmup + T_total]
        target_values = target[warmup : warmup + T_total]

        return {
            "contexts": contexts,
            "scores_matrix": scores_matrix,
            "lookback_windows": lookback_windows,
            "timestamps": timestamps,
            "target_values": target_values,
            "dataset_name": name,
            "freq": config.freq,
        }

    def prepare_raw_series(
        self,
        name: str,
        num_specs: Optional[int] = None,
        warmup_fraction: float = 0.05,
    ) -> Optional[Dict]:
        """
        Return raw series data for external CQR scoring.

        Unlike :meth:`prepare_bandit_experiment`, this does **not** build
        scores or context features -- it just loads the dataset, identifies
        the target column, computes the warmup, and returns the pieces
        that the caller needs to compose CQR scoring with rich contexts.

        Parameters
        ----------
        name : str
            Dataset name (key in REGISTRY).
        num_specs : int, optional
            If given, use only the first ``num_specs`` lookback windows.
        warmup_fraction : float
            Fraction of timesteps to skip at the start.

        Returns
        -------
        dict or None
            ``df``               -- pd.DataFrame (full dataset)
            ``target``           -- np.ndarray, shape (N,) raw target values
            ``lookback_windows`` -- list[int]
            ``config``           -- RealDatasetConfig
            ``warmup``           -- int  (number of steps to skip)
            ``T_total``          -- int  (number of scoreable steps)
        """
        df = self.load(name)
        if df is None:
            return None

        config = self.REGISTRY[name]
        target_col = config.target_col

        # For Electricity we average sampled client columns into a target.
        if name == "Electricity" and target_col not in df.columns:
            target_col = df.columns[0]

        if target_col not in df.columns:
            logger.error(
                "Target column '%s' not found in %s. Columns: %s",
                target_col, name, list(df.columns),
            )
            return None

        target = df[target_col].values.astype(np.float64)

        lookback_windows = list(config.lookback_windows)
        if num_specs is not None:
            lookback_windows = lookback_windows[:num_specs]

        max_lb = max(lookback_windows)
        warmup = max(int(len(target) * warmup_fraction), max_lb + 1)

        T_total = len(target) - warmup
        if T_total <= 0:
            logger.error(
                "Dataset %s is too short for the requested lookback windows.",
                name,
            )
            return None

        return {
            "df": df,
            "target": target,
            "lookback_windows": lookback_windows,
            "config": config,
            "warmup": warmup,
            "T_total": T_total,
        }

    # ------------------------------------------------------------------
    # Private helpers -- downloading
    # ------------------------------------------------------------------

    def _download(self, config: RealDatasetConfig) -> Optional[pd.DataFrame]:
        """Download a single dataset and return a cleaned DataFrame."""
        import urllib.request

        url = config.url
        req = urllib.request.Request(url, headers={"User-Agent": "conformal_ts/1.0"})

        with urllib.request.urlopen(req, timeout=120) as resp:
            raw_bytes = resp.read()

        if config.is_zip:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                member = config.zip_member or zf.namelist()[0]
                with zf.open(member) as f:
                    raw_bytes = f.read()

        if config.is_gzip:
            raw_bytes = gzip.decompress(raw_bytes)

        text = raw_bytes.decode("utf-8", errors="replace")

        # Monash .tsf files don't parse as CSV; handle them before pd.read_csv
        if config.name == "NN5_Daily":
            return self._process_monash_tsf(text, config)

        header_arg = None if config.no_header else config.header
        df = pd.read_csv(
            io.StringIO(text),
            sep=config.separator,
            decimal=config.decimal,
            header=header_arg,
            low_memory=False,
        )

        # Auto-name columns for headerless files
        if config.no_header:
            df.columns = [f"col_{i}" for i in range(len(df.columns))]

        # Post-process per dataset
        if config.name == "Electricity":
            df = self._process_electricity(df, config)
        elif config.name == "EPEX_SPOT":
            df = self._process_epex_spot(df, config)
        elif config.name == "M4_Hourly":
            df = self._process_m4_hourly(df, config)
        elif config.no_header:
            df = self._process_headerless(df, config)
        else:
            df = self._process_generic(df, config)

        return df

    @staticmethod
    def _process_generic(
        df: pd.DataFrame, config: RealDatasetConfig
    ) -> pd.DataFrame:
        """Process ETT-style CSVs (date + numeric columns)."""
        date_col = config.date_col
        if date_col not in df.columns:
            # Guess: first column that looks like a date
            for col in df.columns:
                if "date" in col.lower() or "time" in col.lower():
                    date_col = col
                    break
            else:
                # Fall back to first column
                date_col = df.columns[0]

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()

        # Keep only numeric columns
        df = df.select_dtypes(include=[np.number])

        # Drop rows that are entirely NaN
        df = df.dropna(how="all")

        # Forward-fill small gaps, then drop remaining NaN rows
        df = df.ffill(limit=4).dropna()

        return df

    @staticmethod
    def _process_headerless(
        df: pd.DataFrame, config: RealDatasetConfig
    ) -> pd.DataFrame:
        """Process headerless numeric files (exchange_rate, traffic).

        These have no date column -- we synthesize a DatetimeIndex from
        the configured frequency so downstream code works uniformly.
        """
        df = df.select_dtypes(include=[np.number])
        df = df.dropna(how="all")

        # Synthesize datetime index
        freq_map = {"D": "D", "H": "h", "15min": "15min", "30min": "30min"}
        pd_freq = freq_map.get(config.freq, "h")
        df.index = pd.date_range("2000-01-01", periods=len(df), freq=pd_freq)
        df.index.name = "date"

        return df

    def _process_electricity(
        self, df: pd.DataFrame, config: RealDatasetConfig
    ) -> pd.DataFrame:
        """
        Process the UCI Electricity dataset.

        The raw file uses ``;`` as separator and ``,`` as decimal.
        We sample a few clients, aggregate to reduce noise, and
        set a DatetimeIndex.
        """
        # The first column is the datetime index (unnamed or '')
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()

        # Ensure all columns are numeric (handles any remaining string values)
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(",", ".", regex=False)
                )
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop columns that are all-zero or all-NaN (some clients have no data)
        df = df.loc[:, (df != 0).any(axis=0)]
        df = df.dropna(axis=1, how="all")

        # Sample a handful of clients for tractability
        n_sample = config.sample_clients or 5
        rng = np.random.default_rng(42)
        chosen = rng.choice(df.columns, size=min(n_sample, len(df.columns)), replace=False)
        df = df[chosen].copy()

        # Forward-fill short gaps
        df = df.ffill(limit=4).dropna()

        return df

    @staticmethod
    def _process_epex_spot(
        df: pd.DataFrame, config: RealDatasetConfig
    ) -> pd.DataFrame:
        """Extract French day-ahead price column from the OPSD 250-column panel."""
        ts_col = "utc_timestamp"
        price_col = config.target_col
        if ts_col not in df.columns or price_col not in df.columns:
            raise ValueError(
                f"OPSD schema changed: expected '{ts_col}' and '{price_col}'. "
                f"Got {list(df.columns)[:10]}..."
            )
        df = df[[ts_col, price_col]].copy()
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
        df = df.set_index(ts_col).sort_index()
        df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
        df = df.dropna()
        df.index.name = "date"
        return df

    @staticmethod
    def _process_m4_hourly(
        df: pd.DataFrame, config: RealDatasetConfig
    ) -> pd.DataFrame:
        """Concatenate the first 20 M4 hourly series into a single long series.

        The raw CSV is ragged: column 0 is the series id ('H1','H2',...),
        remaining columns are observations (padded with NaN). We pick the
        first 20 rows, flatten, drop NaNs to obtain ~14k hourly points.
        """
        n_series = 20
        sub = df.iloc[:n_series]
        values_cols = [c for c in sub.columns if c != sub.columns[0]]
        flat = sub[values_cols].to_numpy(dtype=np.float64).reshape(-1)
        flat = flat[~np.isnan(flat)]
        index = pd.date_range("2000-01-01", periods=len(flat), freq="h")
        out = pd.DataFrame({"series": flat}, index=index)
        out.index.name = "date"
        return out

    @staticmethod
    def _process_monash_tsf(
        text: str, config: RealDatasetConfig
    ) -> pd.DataFrame:
        """Parse a Monash .tsf time-series archive file.

        Format: lines starting with '@' are metadata, '@data' marks the start
        of the data section, then each data line is
        'series_name:start_ts:v1,v2,v3,...' (some variants omit fields).
        """
        lines = text.splitlines()
        data_start = None
        for i, line in enumerate(lines):
            if line.strip().lower() == "@data":
                data_start = i + 1
                break
        if data_start is None:
            raise ValueError("No @data marker in tsf file")

        series_list: List[List[float]] = []
        for line in lines[data_start:]:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("@"):
                continue
            parts = line.split(":")
            values_str = parts[-1]
            values = [
                float(v) if v not in ("", "?") else np.nan
                for v in values_str.split(",")
            ]
            series_list.append(values)

        n_series = min(20, len(series_list))
        flat: List[float] = []
        for s in series_list[:n_series]:
            flat.extend(v for v in s if not np.isnan(v))
        arr = np.array(flat, dtype=np.float64)
        index = pd.date_range("2000-01-01", periods=len(arr), freq="D")
        out = pd.DataFrame({"series": arr}, index=index)
        out.index.name = "date"
        return out

    @staticmethod
    def _generate_australian_demand() -> pd.DataFrame:
        """
        Generate a realistic half-hourly Australian electricity demand series.

        The series has:
        - Strong daily and weekly seasonality
        - Temperature-driven summer/winter peaks
        - Stochastic volatility (regime-like behaviour)
        - Occasional demand spikes (heatwaves / cold snaps)
        """
        rng = np.random.default_rng(2024)

        start = pd.Timestamp("2015-01-01")
        periods = 48 * 365 * 3  # 3 years of half-hourly data
        index = pd.date_range(start, periods=periods, freq="30min")

        hour_frac = index.hour + index.minute / 60.0
        day_of_year = index.dayofyear
        day_of_week = index.dayofweek  # 0=Mon

        # -- Base load (MW) ------------------------------------------------
        base_load = 5000.0

        # Daily pattern: morning + evening peaks
        daily = (
            800 * np.exp(-((hour_frac - 8.5) ** 2) / 4)
            + 1200 * np.exp(-((hour_frac - 18.5) ** 2) / 3)
            - 600 * np.exp(-((hour_frac - 3.0) ** 2) / 6)
        )

        # Weekly: lower on weekends
        weekend_factor = np.where(day_of_week >= 5, -500.0, 0.0)

        # Seasonal: summer + winter heating/cooling peaks (southern hemisphere)
        seasonal = (
            1000 * np.cos(2 * np.pi * (day_of_year - 15) / 365)   # Summer peak Jan
            + 400 * np.cos(2 * np.pi * (day_of_year - 195) / 365)  # Winter peak Jul
        )

        # Temperature proxy (drives volatility)
        temperature = (
            25 + 12 * np.sin(2 * np.pi * (day_of_year - 15) / 365)
            + 5 * np.sin(2 * np.pi * hour_frac / 24)
            + rng.normal(0, 2, size=periods)
        )

        # Temperature-demand interaction (cooling above 30, heating below 10)
        temp_effect = np.where(
            temperature > 30,
            150 * (temperature - 30),
            np.where(temperature < 10, 100 * (10 - temperature), 0.0),
        )

        # Random walk for slow-moving demand level
        level_noise = np.cumsum(rng.normal(0, 5, size=periods))
        level_noise -= np.mean(level_noise)

        # Occasional spikes (heatwaves etc.)
        spike_mask = rng.random(periods) < 0.002
        spikes = spike_mask * rng.uniform(500, 2000, size=periods)

        demand = (
            base_load
            + daily
            + weekend_factor
            + seasonal
            + temp_effect
            + level_noise
            + spikes
            + rng.normal(0, 150, size=periods)
        )
        demand = np.maximum(demand, 1000)  # Floor

        df = pd.DataFrame(
            {"demand": demand, "temperature": temperature},
            index=index,
        )
        return df

    @staticmethod
    def _generate_varbreak(seed: int = 7) -> pd.DataFrame:
        """Piecewise noise-scale synthetic with known regime structure.

        Hourly series: level + diurnal sinusoid + AR(1) innovations whose
        standard deviation switches abruptly between a calm regime
        (sigma = 0.5) and a volatile regime (sigma = 2.5), with segment
        lengths drawn uniformly from [250, 450] steps.  Immediately after
        a break, short CQR calibration windows adapt faster (long windows
        are contaminated by the stale regime); deep inside a segment,
        long windows are less noisy.  The optimal calibration window
        therefore provably shifts over time -- the exact phenomenon
        calibration-window selection targets.  Fixed seed: the series is
        treated as a fixed dataset; per-seed randomness comes from the
        experiment pipeline.
        """
        rng = np.random.default_rng(seed)
        T_total = 3000
        sigmas = (0.5, 2.5)
        regime = 0
        noise_sd = np.empty(T_total)
        t = 0
        while t < T_total:
            seg = int(rng.integers(250, 451))
            noise_sd[t:t + seg] = sigmas[regime]
            regime = 1 - regime
            t += seg
        x = 0.0
        vals = np.empty(T_total)
        for i in range(T_total):
            x = 0.7 * x + rng.normal(0.0, noise_sd[i])
            vals[i] = 20.0 + x
        index = pd.date_range("2023-01-01", periods=T_total, freq="h")
        return pd.DataFrame({"y": vals}, index=index)

    @staticmethod
    def _load_gefcom_real(track: str, zone: int = 1) -> Optional[pd.DataFrame]:
        """Load the real GEFCom2014 wind or solar track, single zone.

        Expects the long-format track files (TIMESTAMP + target + weather
        + ZONE_k one-hot columns) at ``./data/gefcom2014/{wind,solar}.csv``.
        Returns a single-column ``"power"`` DataFrame with an hourly
        DatetimeIndex -- the same shape the synthetic generator returns --
        or ``None`` when the files are absent.
        """
        from pathlib import Path as _Path
        import os
        # Resolve independently of the working directory. A cwd-relative
        # path silently yields the synthetic fallback when the caller runs
        # from elsewhere, which is how a synthetic series once reached a
        # results table labelled as real.
        candidates = []
        env = os.environ.get("CIS_DATA_DIR")
        if env:
            candidates.append(_Path(env) / "gefcom2014")
        candidates.append(_Path(__file__).resolve().parents[2] / "data" / "gefcom2014")
        candidates.append(_Path("./data/gefcom2014").resolve())
        path = None
        for c in candidates:
            if (c / f"{track}.csv").exists():
                path = c / f"{track}.csv"
                break
        if path is None:
            logger.warning("GEFCom %s: no real data found; searched %s",
                           track, ", ".join(str(c) for c in candidates))
            return None
        try:
            raw = pd.read_csv(path, parse_dates=["TIMESTAMP"])
            zone_col = f"ZONE_{zone}"
            target_col = "TARGETVAR" if track == "wind" else "POWER"
            if zone_col not in raw.columns or target_col not in raw.columns:
                logger.warning("GEFCom %s: unexpected columns in %s", track, path)
                return None
            sel = raw[raw[zone_col] == 1].copy()
            sel = sel.dropna(subset=[target_col])
            sel = sel.sort_values("TIMESTAMP").set_index("TIMESTAMP")
            # Power is normalized to [0, 1] in the competition files; scale
            # to [0, 100] so interval scores are on a readable scale.
            df = pd.DataFrame({"power": 100.0 * sel[target_col].values},
                              index=pd.DatetimeIndex(sel.index))
            return df
        except Exception as exc:  # pragma: no cover -- defensive
            logger.warning("GEFCom %s: failed to load real data (%s)", track, exc)
            return None

    @staticmethod
    def _generate_gefcom(track: str, seed: int = 2024) -> pd.DataFrame:
        """Generate a GEFCom2014-style hourly wind or solar power series.

        Wraps :meth:`conformal_ts.data.gefcom2014.GEFCom2014Loader.create_synthetic_gefcom_data`
        and returns a single-column ``"power"`` DataFrame indexed by
        an hourly DatetimeIndex, so the series can be consumed by the
        shared CQR pipeline exactly like the other real datasets.
        """
        from conformal_ts.data.gefcom2014 import (
            GEFCom2014Loader, GEFComConfig,
        )
        gef = GEFCom2014Loader(GEFComConfig(track=track))
        raw = gef.create_synthetic_gefcom_data(
            num_zones=1, num_days=365 * 2, seed=seed,
        )
        zone_col = [c for c in raw.columns if c.startswith("zone")][0]
        df = pd.DataFrame({"power": raw[zone_col].values}, index=raw.index)
        df.index = pd.DatetimeIndex(df.index)  # ensure DatetimeIndex
        return df

    # ------------------------------------------------------------------
    # Private helpers -- context features
    # ------------------------------------------------------------------

    @staticmethod
    def build_context_features(
        df: pd.DataFrame,
        target: np.ndarray,
        warmup: int,
        T: int,
        config: RealDatasetConfig,
    ) -> np.ndarray:
        """
        Build the (T, D) feature matrix used as bandit context.

        Features:
            0. bias (constant 1)
            1. recent mean      (last 24 native steps, or 24)
            2. recent std       (last 24 native steps)
            3. recent trend     (mean last 12 - mean prev 12)
            4. longer mean      (last 168 native steps, capped)
            5. longer std       (last 168 native steps, capped)
            6. hour-of-day sin  (if datetime index)
            7. hour-of-day cos
            8. day-of-week sin
            9. day-of-week cos
           10. lag 1
           11. lag 2
           12. lag 3
        """
        # Choose short and long context spans proportional to frequency
        freq_map = {"H": 1, "15min": 4, "30min": 2, "D": 1 / 24}
        steps_per_hour = freq_map.get(config.freq, 1)
        short_span = max(int(24 * steps_per_hour), 4)
        long_span = max(int(168 * steps_per_hour), short_span + 1)

        D = 13
        contexts = np.zeros((T, D), dtype=np.float32)

        idx = df.index
        has_datetime = hasattr(idx, "hour")

        for i in range(T):
            t = warmup + i

            # Bias
            contexts[i, 0] = 1.0

            # Short window statistics
            start_s = max(0, t - short_span)
            seg_short = target[start_s:t]
            if len(seg_short) >= 2:
                contexts[i, 1] = np.mean(seg_short)
                contexts[i, 2] = np.std(seg_short, ddof=1)
                half = len(seg_short) // 2
                contexts[i, 3] = np.mean(seg_short[half:]) - np.mean(seg_short[:half])
            else:
                contexts[i, 1] = target[t - 1] if t > 0 else 0.0
                contexts[i, 2] = 1e-4

            # Long window statistics
            start_l = max(0, t - long_span)
            seg_long = target[start_l:t]
            if len(seg_long) >= 2:
                contexts[i, 4] = np.mean(seg_long)
                contexts[i, 5] = np.std(seg_long, ddof=1)

            # Calendar features
            if has_datetime and t < len(idx):
                ts = idx[t]
                # Hour-of-day only for sub-daily frequencies;
                # for daily+ data hour is always 0 → constant features
                # (col 7 = cos(0) = 1 is collinear with bias).
                if steps_per_hour >= 1:
                    hour_frac = ts.hour + ts.minute / 60.0
                    contexts[i, 6] = np.sin(2 * np.pi * hour_frac / 24)
                    contexts[i, 7] = np.cos(2 * np.pi * hour_frac / 24)
                contexts[i, 8] = np.sin(2 * np.pi * ts.dayofweek / 7)
                contexts[i, 9] = np.cos(2 * np.pi * ts.dayofweek / 7)

            # Lag features
            if t >= 1:
                contexts[i, 10] = target[t - 1]
            if t >= 2:
                contexts[i, 11] = target[t - 2]
            if t >= 3:
                contexts[i, 12] = target[t - 3]

        # Standardise each column to zero-mean, unit-variance (except bias)
        for d in range(1, D):
            col = contexts[:, d]
            mu = np.mean(col)
            sigma = np.std(col)
            if sigma > 1e-8:
                contexts[:, d] = (col - mu) / sigma

        return contexts


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------

def load_real_dataset(name: str, cache_dir: str = "./cache/datasets") -> Optional[pd.DataFrame]:
    """Shortcut to load a dataset without explicitly creating a loader."""
    return RealDatasetLoader(cache_dir=cache_dir).load(name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    loader = RealDatasetLoader()

    print("Available datasets:", loader.list_available())
    print()

    for ds_name in loader.list_available():
        print(f"--- {ds_name} ---")
        df = loader.load(ds_name)
        if df is not None:
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns[:8])}{'...' if len(df.columns) > 8 else ''}")
            print(f"  Date range: {df.index.min()} to {df.index.max()}")
        else:
            print("  (download failed)")
        print()

    # Quick bandit-experiment test on the cheapest dataset
    print("=== Bandit experiment (AustralianElecDemand) ===")
    result = loader.prepare_bandit_experiment("AustralianElecDemand", num_specs=4)
    if result is not None:
        print(f"  contexts shape:      {result['contexts'].shape}")
        print(f"  scores_matrix shape: {result['scores_matrix'].shape}")
        print(f"  lookback windows:    {result['lookback_windows']}")
        print(f"  target_values shape: {result['target_values'].shape}")
        # Sanity: mean score per spec
        mean_scores = np.nanmean(result["scores_matrix"], axis=0)
        for w, s in zip(result["lookback_windows"], mean_scores):
            print(f"    window={w:>5d}  mean_IS={s:.2f}")
