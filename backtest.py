"""
backtest.py
P2-ETF-VLSTM-SIGNAL

Backtest engine and live signal generator.
FIXED: stream_consensus() now returns fixed-schema dict to prevent HF casting errors.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any


def execute_strategy(preds, probs, returns, dates, target_etfs, fee_bps=10.0):
    """
    Simulate strategy: hold ETF with highest predicted probability.
    preds: array of int indices
    probs: array of probability distributions
    returns: array of next-day returns (float)
    dates: array of dates
    target_etfs: list of ETF tickers
    fee_bps: transaction cost in basis points
    """
    n = len(preds)
    daily_rets = []
    etf_held = []

    for i in range(n):
        held_idx = preds[i]
        ret = returns[i, held_idx]

        # Transaction cost on turnover (simplified: cost every day)
        cost = 2 * fee_bps / 10000  # round-trip
        net_ret = ret - cost

        daily_rets.append(net_ret)
        etf_held.append(target_etfs[held_idx])

    daily_rets = np.array(daily_rets)

    # Metrics
    ann_return = np.mean(daily_rets) * 252
    ann_vol = np.std(daily_rets) * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # Max drawdown
    cum = np.cumprod(1 + daily_rets)
    running_max = np.maximum.accumulate(cum)
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()

    # Hit rate
    hit_rate = np.mean(daily_rets > 0)

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "hit_rate": hit_rate,
        "daily_rets": daily_rets,
        "etf_held": etf_held,
        "dates": dates,
    }


def generate_live_signal(model, X_full, scale_mean, scale_std, lookback, target_etfs, feat_names):
    """
    Generate signal for today using the most recent lookback days.
    """
    from vlstm import predict

    if X_full is None or len(X_full) < lookback:
        return {"signal": "N/A", "probabilities": {}, "confidence": 0}

    # Get last lookback days
    X_recent = X_full[-lookback:]
    X_scaled = (X_recent - scale_mean) / scale_std
    X_seq = X_scaled.reshape(1, lookback, -1)

    # Predict
    preds, probs, attn = predict(model, X_seq)

    signal_idx = preds[0]
    signal = target_etfs[signal_idx]
    probabilities = {target_etfs[i]: float(probs[0, i]) for i in range(len(target_etfs))}
    confidence = float(probs[0, signal_idx])

    return {
        "signal": signal,
        "probabilities": probabilities,
        "confidence": confidence,
        "top_features": [],  # Placeholder for VSN attention
    }


def summarise_window_result(window, train_res, backtest, live_signal, lookback, option):
    """
    Summarise all results for one window into a dict.
    """
    return {
        "window_label": window["label"],
        "window_start": window["start_year"],
        "window_end": window["end_year"],
        "stream": window.get("stream", "unknown"),
        "option": option,
        "lookback": lookback,
        "val_sharpe": train_res.get("val_sharpe", 0),
        "train_sharpe": train_res.get("train_sharpe", 0),
        "epochs_run": train_res.get("epochs_run", 0),
        "best_epoch": train_res.get("best_epoch", 0),
        "backtest_ann_return": backtest.get("ann_return", 0),
        "backtest_sharpe": backtest.get("sharpe", 0),
        "backtest_max_dd": backtest.get("max_dd", 0),
        "backtest_hit_rate": backtest.get("hit_rate", 0),
        "live_signal": live_signal.get("signal", "N/A"),
        "live_confidence": live_signal.get("confidence", 0),
        "live_probabilities": live_signal.get("probabilities", {}),
    }


def stream_consensus(results: List[Dict], target_etfs: List[str]) -> Dict[str, Any]:
    """
    Compute consensus across all windows in a stream.

    FIXED: Returns fixed-schema dict to prevent HF dataset casting errors.
    Instead of {"GDX": 5, "XLV": 3, ...} we return:
    {
        "signal": "GDX",
        "strength": "high",
        "score_pct": 75.0,
        "agreement": 60.0,
        "votes": {"GDX": 5, "XLV": 3, ...},  # Nested dict for ETF votes
        "z_score": 2.1,
    }
    """
    if not results:
        return {
            "signal": "N/A",
            "strength": "none",
            "score_pct": 0,
            "agreement": 0,
            "votes": {},
            "z_score": 0,
        }

    # Count votes per ETF
    votes = {etf: 0 for etf in target_etfs}
    for r in results:
        signal = r.get("live_signal", "N/A")
        if signal in votes:
            votes[signal] += 1

    total = len(results)
    winner = max(votes, key=votes.get)
    winner_count = votes[winner]

    # Calculate agreement percentage
    agreement = (winner_count / total) * 100 if total > 0 else 0

    # Calculate z-score for conviction
    vote_counts = list(votes.values())
    mean_votes = np.mean(vote_counts)
    std_votes = np.std(vote_counts)
    z_score = (winner_count - mean_votes) / std_votes if std_votes > 0 else 0

    # Determine strength
    if agreement >= 60 and z_score > 1.5:
        strength = "high"
    elif agreement >= 40:
        strength = "moderate"
    else:
        strength = "low"

    # Score percentage (0-100)
    score_pct = agreement

    # FIX: Return fixed schema with votes nested
    return {
        "signal": winner,
        "strength": strength,
        "score_pct": round(score_pct, 1),
        "agreement": round(agreement, 1),
        "votes": votes,  # Nested dict prevents schema conflicts
        "z_score": round(float(z_score), 2),
        "total_windows": total,
    }
