"""
loader.py
P2-ETF-VLSTM-SIGNAL

Loads master_data.parquet from HuggingFace dataset and computes all derived
features following the Oxford paper (Saly-Kaufmann et al. 2026):

Feature engineering per ETF (Close-derived):
  - Normalised returns: 1d, 5d, 21d, 63d, 126d, 252d  (vol-scaled)
  - MACD signals:       (4,12), (8,24), (32,96)        (double-normalised)
  - EWMA volatility:    span=63
  - Vol-scaling factor: 1/sigma

Option A: ETF-derived features + macro (VIX, DXY, T10Y2Y, TBILL_3M,
                                         IG_SPREAD, HY_SPREAD)
Option B: ETF-derived features only

Target: for each row, which ETF had the highest next-day return
        (argmax over 1-day forward returns of the 6 target ETFs)
"""

import numpy as np
import pandas as pd
from datasets import load_dataset

# ── Constants ─────────────────────────────────────────────────────────────────

TARGET_ETFS   = ["TLT", "VNQ", "SLV", "GLD", "HYG", "LQD"]
BENCH_COLS    = ["SPY", "AGG"]                          # features, not targets
MACRO_COLS    = ["VIX", "DXY", "T10Y2Y", "TBILL_3M", "IG_SPREAD", "HY_SPREAD"]
PRICE_COLS    = TARGET_ETFS + BENCH_COLS                # all price series

RETURN_HORIZONS = [1, 5, 21, 63, 126, 252]             # trading days
MACD_PAIRS      = [(4, 12), (8, 24), (32, 96)]         # (fast, slow)
EWMA_VOL_SPAN   = 63
MIN_ROWS        = 300                                   # safety floor


# ── HuggingFace loader ────────────────────────────────────────────────────────

def load_raw(hf_token: str) -> pd.DataFrame:
    """
    Load master_data.parquet from HuggingFace.
    Returns a DataFrame indexed by date, sorted ascending.
    """
    ds = load_dataset(
        "P2SAMAPA/fi-etf-macro-signal-master-data",
        data_files="master_data.parquet",
        split="train",
        token=hf_token,
    )
    df = ds.to_pandas()

    # Identify and set the date index
    date_col = "__index_level_0__"
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    df.index.name = "date"
    df = df.sort_index()

    # Keep only columns we know about; drop anything unexpected
    keep = [c for c in PRICE_COLS + MACRO_COLS if c in df.columns]
    df = df[keep].copy()

    return df


# ── Low-level feature helpers ─────────────────────────────────────────────────

def _ewma_vol(prices: pd.Series, span: int = EWMA_VOL_SPAN) -> pd.Series:
    """
    EWMA volatility of daily returns using pandas ewm.
    Returns annualised daily vol (not annualised — kept daily for normalisation).
    """
    rets  = prices.pct_change()
    vol   = rets.ewm(span=span, min_periods=span // 2).std()
    return vol.replace(0, np.nan)


def _normalised_return(prices: pd.Series, h: int, vol: pd.Series) -> pd.Series:
    """
    Vol-normalised h-day return:  r_norm = (P_t / P_{t-h} - 1) / (sigma * sqrt(h))
    """
    raw = prices.pct_change(h)
    return raw / (vol * np.sqrt(h))


def _macd_signal(prices: pd.Series, fast: int, slow: int) -> pd.Series:
    """
    Double-normalised MACD signal following the paper (eqs 19-21):
      MACD_t  = EWMA(fast) - EWMA(slow)
      q_t     = MACD_t / Std_63(Price)
      Signal  = q_t / Std_252(q_t)
    """
    lam_f   = 2.0 / (fast + 1)
    lam_s   = 2.0 / (slow + 1)

    ema_f   = prices.ewm(alpha=lam_f, adjust=False).mean()
    ema_s   = prices.ewm(alpha=lam_s, adjust=False).mean()
    macd    = ema_f - ema_s

    std63   = prices.rolling(63, min_periods=32).std().replace(0, np.nan)
    q       = macd / std63

    std252q = q.rolling(252, min_periods=126).std().replace(0, np.nan)
    signal  = q / std252q

    return signal


# ── Per-ticker feature builder ────────────────────────────────────────────────

def _features_for_ticker(prices: pd.Series, prefix: str) -> pd.DataFrame:
    """
    Build all Close-derived features for a single price series.
    Returns a DataFrame with columns prefixed by `prefix`.
    """
    vol    = _ewma_vol(prices)
    cols   = {}

    # Vol-normalised returns
    for h in RETURN_HORIZONS:
        cols[f"{prefix}_r{h}d"] = _normalised_return(prices, h, vol)

    # MACD signals
    for fast, slow in MACD_PAIRS:
        cols[f"{prefix}_macd_{fast}_{slow}"] = _macd_signal(prices, fast, slow)

    # EWMA vol level (normalised by its own rolling mean to make it stationary)
    vol_norm = vol / vol.rolling(252, min_periods=126).mean()
    cols[f"{prefix}_vol"] = vol_norm

    # Vol-scaling factor (clipped to avoid extremes)
    vs = (1.0 / vol).clip(upper=400)
    cols[f"{prefix}_vsfactor"] = vs

    return pd.DataFrame(cols)


# ── Macro normaliser ──────────────────────────────────────────────────────────

def _normalise_macro(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score normalise each macro column using its own expanding mean/std
    (to avoid lookahead bias).
    Returns a DataFrame of normalised macro columns.
    """
    macro = df_raw[[c for c in MACRO_COLS if c in df_raw.columns]].copy()
    normed = {}
    for col in macro.columns:
        m   = macro[col].expanding(min_periods=63).mean()
        s   = macro[col].expanding(min_periods=63).std().replace(0, np.nan)
        normed[col] = ((macro[col] - m) / s).clip(-5, 5)
    return pd.DataFrame(normed, index=df_raw.index)


# ── Target builder ────────────────────────────────────────────────────────────

def _build_targets(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    For each row t, compute next-day returns for all target ETFs.
    Target label = index of ETF with highest next-day return (argmax).
    Also return the raw next-day return matrix for Sharpe loss training.
    """
    fwd_rets = {}
    for etf in TARGET_ETFS:
        if etf in df_raw.columns:
            fwd_rets[etf] = df_raw[etf].pct_change().shift(-1)   # next-day return

    fwd_df     = pd.DataFrame(fwd_rets)
    label      = fwd_df.values.argmax(axis=1).astype(np.int64)
    label_sr   = pd.Series(label, index=df_raw.index, name="target_label")

    return fwd_df, label_sr


# ── Main feature engineering entry point ─────────────────────────────────────

def build_features(
    df_raw: pd.DataFrame,
    option: str = "A",          # "A" = with macro, "B" = without macro
    start_year: int = 2008,
    end_year:   int = None,     # None = use all available data
) -> dict:
    """
    Full feature engineering pipeline.

    Parameters
    ----------
    df_raw     : raw DataFrame from load_raw()
    option     : "A" (Close-derived + macro) or "B" (Close-derived only)
    start_year : first year to include in training slice
    end_year   : last year to include (inclusive); None = latest available

    Returns
    -------
    dict with keys:
        X              : np.ndarray [T, n_features]  float32
        y_labels       : np.ndarray [T]              int64  (argmax ETF index)
        y_returns      : np.ndarray [T, n_etfs]      float32 (next-day returns)
        feature_names  : list[str]
        dates          : pd.DatetimeIndex
        target_etfs    : list[str]
        n_etfs         : int
    """
    assert option in ("A", "B"), "option must be 'A' or 'B'"

    # ── Slice by year ─────────────────────────────────────────────────────────
    df = df_raw[df_raw.index.year >= start_year].copy()
    if end_year is not None:
        df = df[df.index.year <= end_year]

    if len(df) < MIN_ROWS:
        raise ValueError(
            f"Only {len(df)} rows for {start_year}–{end_year}. "
            f"Need at least {MIN_ROWS}."
        )

    # ── Build Close-derived features for every price series ──────────────────
    feature_frames = []
    for ticker in PRICE_COLS:
        if ticker in df.columns:
            feature_frames.append(
                _features_for_ticker(df[ticker], prefix=ticker)
            )

    feat_df = pd.concat(feature_frames, axis=1)

    # ── Optionally add macro ──────────────────────────────────────────────────
    if option == "A":
        macro_df = _normalise_macro(df)
        feat_df  = pd.concat([feat_df, macro_df], axis=1)

    # ── Build targets ─────────────────────────────────────────────────────────
    fwd_returns, labels = _build_targets(df)

    # ── Align all DataFrames on the same index ────────────────────────────────
    combined = feat_df.join(fwd_returns, how="inner").join(labels, how="inner")
    combined = combined.dropna(subset=[labels.name])   # drop rows with no label

    # Drop warmup rows where features are still NaN
    # (longest warmup = 252-day rolling std in MACD double-normalisation)
    combined = combined.dropna(thresh=int(len(combined.columns) * 0.85))

    feature_names = [c for c in combined.columns
                     if c not in list(fwd_returns.columns) + [labels.name]]
    ret_cols      = list(fwd_returns.columns)

    X        = combined[feature_names].values.astype(np.float32)
    y_labels = combined[labels.name].values.astype(np.int64)
    y_returns= combined[ret_cols].values.astype(np.float32)

    # Final NaN imputation (column mean) — safety net
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            col_mean = np.nanmean(X[:, j])
            X[mask, j] = col_mean if not np.isnan(col_mean) else 0.0

    # Clip extreme values to [-10, 10] (paper clips vol-scaled returns)
    X = np.clip(X, -10.0, 10.0)

    return {
        "X":             X,
        "y_labels":      y_labels,
        "y_returns":     y_returns,
        "feature_names": feature_names,
        "dates":         combined.index,
        "target_etfs":   TARGET_ETFS,
        "n_etfs":        len(TARGET_ETFS),
    }


# ── Window generators ─────────────────────────────────────────────────────────

def expanding_windows(
    first_train_end: int = 2011,
    data_end_year:   int = None,
    df_raw:          pd.DataFrame = None,
    step:            int = 3,
) -> list:
    """
    Generates expanding window configs in step-year increments:
      2008→2011, 2008→2014, 2008→2017, 2008→2020, 2008→2023, 2008→2026

    The final window (data_end_year) is always included even if not on step boundary.
    Returns list of dicts: {start_year, end_year, label, stream}
    """
    if data_end_year is None:
        data_end_year = df_raw.index.year.max()

    ends = list(range(first_train_end, data_end_year, step))
    if data_end_year not in ends:
        ends.append(data_end_year)

    windows = []
    for end in ends:
        windows.append({
            "start_year": 2008,
            "end_year":   end,
            "label":      f"2008→{end}",
            "stream":     "expanding",
        })
    return windows


def shrinking_windows(
    data_end_year: int = None,
    df_raw:        pd.DataFrame = None,
    step:          int = 3,
) -> list:
    """
    Generates shrinking window configs in step-year increments:
      2008→2026, 2011→2026, 2014→2026, 2017→2026, 2020→2026, 2023→2026

    The final window (data_end_year-1 start) is always included.
    Returns list of dicts: {start_year, end_year, label, stream}
    """
    if data_end_year is None:
        data_end_year = df_raw.index.year.max()

    starts = list(range(2008, data_end_year, step))
    if (data_end_year - 1) not in starts:
        starts.append(data_end_year - 1)

    windows = []
    for start in starts:
        windows.append({
            "start_year": start,
            "end_year":   data_end_year,
            "label":      f"{start}→{data_end_year}",
            "stream":     "shrinking",
        })
    return windows


def all_windows(df_raw: pd.DataFrame, step: int = 3) -> list:
    """Returns all expanding + shrinking windows at given step size."""
    data_end = df_raw.index.year.max()
    exp = expanding_windows(first_train_end=2011, data_end_year=data_end,
                            df_raw=df_raw, step=step)
    shr = shrinking_windows(data_end_year=data_end, df_raw=df_raw, step=step)
    return exp + shr


# ── Train/val/test split ──────────────────────────────────────────────────────

def chronological_split(
    X: np.ndarray,
    y_labels: np.ndarray,
    y_returns: np.ndarray,
    dates: pd.DatetimeIndex,
    train_pct: float = 0.70,
    val_pct:   float = 0.15,
) -> dict:
    """
    Strict chronological split — no shuffling.
    Returns dict of train/val/test arrays.
    """
    n        = len(X)
    t_end    = int(n * train_pct)
    v_end    = int(n * (train_pct + val_pct))

    return {
        "X_train":    X[:t_end],
        "X_val":      X[t_end:v_end],
        "X_test":     X[v_end:],
        "y_train_l":  y_labels[:t_end],
        "y_val_l":    y_labels[t_end:v_end],
        "y_test_l":   y_labels[v_end:],
        "y_train_r":  y_returns[:t_end],
        "y_val_r":    y_returns[t_end:v_end],
        "y_test_r":   y_returns[v_end:],
        "dates_train": dates[:t_end],
        "dates_val":   dates[t_end:v_end],
        "dates_test":  dates[v_end:],
    }


# ── Dataset summary ───────────────────────────────────────────────────────────

def dataset_summary(df_raw: pd.DataFrame) -> dict:
    return {
        "rows":       len(df_raw),
        "start_date": str(df_raw.index[0].date()),
        "end_date":   str(df_raw.index[-1].date()),
        "etfs":       [c for c in TARGET_ETFS if c in df_raw.columns],
        "macro":      [c for c in MACRO_COLS  if c in df_raw.columns],
        "benchmarks": [c for c in BENCH_COLS  if c in df_raw.columns],
    }
