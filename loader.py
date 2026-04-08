"""
loader.py
P2-ETF-VLSTM-SIGNAL

Data loading and feature engineering.
FIXED: Uses input_filename from config to support master_data.parquet
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from huggingface_hub import hf_hub_download

DEFAULT_MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M", "IG_SPREAD", "HY_SPREAD"]


def load_raw_from_dataset(dataset_name: str, hf_token: str, filename: str = "master_data.parquet") -> pd.DataFrame:
    """
    Load raw data from specified HuggingFace dataset.

    FIXED: Now accepts filename parameter to handle master_data.parquet
    """
    try:
        local = hf_hub_download(
            repo_id=dataset_name,
            filename=filename,
            repo_type="dataset",
            token=hf_token,
        )
        df = pd.read_parquet(local)

        # Ensure index is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        elif df.index.name != 'date' and not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        return df.sort_index()
    except Exception as e:
        raise ValueError(f"Failed to load dataset {dataset_name}/{filename}: {e}")


def load_raw(hf_token: str, dataset_name: str = "P2SAMAPA/fi-etf-macro-signal-master-data") -> pd.DataFrame:
    """
    Legacy function - loads FI dataset by default.
    Use load_raw_from_dataset() for new code.
    """
    return load_raw_from_dataset(dataset_name, hf_token, filename="master_data.parquet")


def dataset_summary(df: pd.DataFrame, target_etfs: List[str]) -> dict:
    """Generate summary of loaded dataset."""
    macro_cols = [c for c in df.columns if c in DEFAULT_MACRO_COLS]

    return {
        "rows": len(df),
        "start_date": str(df.index[0].date()),
        "end_date": str(df.index[-1].date()),
        "etfs": [c for c in df.columns if c in target_etfs],
        "macro": macro_cols,
    }


def build_features(df_raw: pd.DataFrame, 
                  option: str = "A",
                  start_year: int = 2008,
                  end_year: Optional[int] = None,
                  target_etfs: List[str] = None,
                  feature_tickers: List[str] = None,
                  macro_cols: List[str] = None) -> dict:
    """
    Build feature matrix X and target arrays y.

    Option A: Close-derived + macro features
    Option B: Close-derived only
    """
    if target_etfs is None:
        target_etfs = ["TLT", "VNQ", "SLV", "GLD", "HYG", "LQD"]
    if feature_tickers is None:
        feature_tickers = target_etfs + ["SPY", "AGG"]
    if macro_cols is None:
        macro_cols = DEFAULT_MACRO_COLS

    # Filter by date range
    mask = df_raw.index.year >= start_year
    if end_year:
        mask &= df_raw.index.year <= end_year
    df = df_raw[mask].copy()

    if len(df) < 100:
        raise ValueError(f"Insufficient data: {len(df)} rows for range {start_year}-{end_year}")

    # Build features for each ticker
    features_list = []
    feature_names = []

    for ticker in feature_tickers:
        close_col = f"{ticker}_close"
        if close_col not in df.columns:
            continue

        close = df[close_col]

        # Returns (vol-normalized)
        for days in [1, 5, 21, 63, 126, 252]:
            if len(close) > days:
                ret = close.pct_change(days)
                vol = close.pct_change(1).rolling(252).std() * np.sqrt(252)
                feat = ret / vol
                features_list.append(feat)
                feature_names.append(f"{ticker}_ret_{days}d")

        # MACD signals
        for fast, slow in [(4, 12), (8, 24), (32, 96)]:
            if len(close) > slow:
                ema_fast = close.ewm(span=fast).mean()
                ema_slow = close.ewm(span=slow).mean()
                macd = (ema_fast - ema_slow) / close.rolling(252).std()
                features_list.append(macd)
                feature_names.append(f"{ticker}_macd_{fast}_{slow}")

        # EWMA volatility
        if len(close) > 63:
            vol = close.pct_change(1).ewm(span=63).std()
            mean_vol = close.pct_change(1).rolling(252).std()
            feat = vol / mean_vol
            features_list.append(feat)
            feature_names.append(f"{ticker}_vol_63")

        # Vol scaling factor
        vol_252 = close.pct_change(1).rolling(252).std() * np.sqrt(252)
        scale = 1.0 / vol_252
        scale = scale.clip(upper=400)
        features_list.append(scale)
        feature_names.append(f"{ticker}_scale")

    # Add macro features for Option A
    if option == "A":
        for col in macro_cols:
            if col in df.columns:
                features_list.append(df[col])
                feature_names.append(col)

    # Stack features
    X = np.column_stack([f.values for f in features_list])

    # Build targets (next-day returns for each target ETF)
    y_returns = []
    for etf in target_etfs:
        close_col = f"{etf}_close"
        if close_col not in df.columns:
            raise ValueError(f"Target ETF {etf} not found in dataset (looking for {close_col})")

        next_ret = df[close_col].pct_change(1).shift(-1)
        y_returns.append(next_ret.values)

    y_returns = np.column_stack(y_returns)

    # Labels: which ETF has best next-day return
    y_labels = np.argmax(y_returns, axis=1)

    # Remove rows with NaN
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y_returns).any(axis=1)
    X = X[valid_mask]
    y_labels = y_labels[valid_mask]
    y_returns = y_returns[valid_mask]
    dates = df.index[valid_mask]

    return {
        "X": X,
        "y_labels": y_labels,
        "y_returns": y_returns,
        "dates": dates,
        "feature_names": feature_names,
    }


def expanding_windows(first_train_end: int = 2011, 
                     data_end_year: int = 2025,
                     df_raw: pd.DataFrame = None,
                     step: int = 3) -> List[dict]:
    """Generate expanding window configurations."""
    windows = []
    for end_year in range(first_train_end, data_end_year + 1, step):
        windows.append({
            "start_year": 2008,
            "end_year": end_year,
            "label": f"2008→{end_year}",
            "stream": "expanding",
        })
    return windows


def shrinking_windows(data_end_year: int = 2025,
                     df_raw: pd.DataFrame = None,
                     step: int = 3) -> List[dict]:
    """Generate shrinking window configurations."""
    windows = []
    for start_year in range(2008, data_end_year, step):
        windows.append({
            "start_year": start_year,
            "end_year": data_end_year,
            "label": f"{start_year}→{data_end_year}",
            "stream": "shrinking",
        })
    return windows


def all_windows(df_raw: pd.DataFrame, step: int = 3) -> Tuple[List[dict], List[dict]]:
    """Generate both expanding and shrinking windows."""
    data_end_year = df_raw.index.year.max()
    exp = expanding_windows(data_end_year=data_end_year, df_raw=df_raw, step=step)
    shr = shrinking_windows(data_end_year=data_end_year, df_raw=df_raw, step=step)
    return exp, shr


def chronological_split(X, y_labels, y_returns, train_pct=0.7, val_pct=0.15):
    """Split data chronologically."""
    n = len(X)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    return {
        "X_train": X[:train_end],
        "X_val": X[train_end:val_end],
        "X_test": X[val_end:],
        "y_train_labels": y_labels[:train_end],
        "y_val_labels": y_labels[train_end:val_end],
        "y_test_labels": y_labels[val_end:],
        "y_train_returns": y_returns[:train_end],
        "y_val_returns": y_returns[train_end:val_end],
        "y_test_returns": y_returns[val_end:],
    }
