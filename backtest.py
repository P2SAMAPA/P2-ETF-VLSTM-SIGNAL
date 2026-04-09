"""
backtest.py
P2-ETF-VLSTM-SIGNAL
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any


def execute_strategy(preds, probs, returns, dates, target_etfs, fee_bps=10.0):
    """Simulate strategy: hold ETF with highest predicted probability."""
    n = len(preds)
    daily_rets = []
    etf_held = []
    
    for i in range(n):
        held_idx = preds[i]
        ret = returns[i, held_idx]
        cost = 2 * fee_bps / 10000
        net_ret = ret - cost
        daily_rets.append(net_ret)
        etf_held.append(target_etfs[held_idx])
    
    daily_rets = np.array(daily_rets)
    
    ann_return = np.mean(daily_rets) * 252
    ann_vol = np.std(daily_rets) * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    
    cum = np.cumprod(1 + daily_rets)
    running_max = np.maximum.accumulate(cum)
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()
    
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
    """Generate signal for today using the most recent lookback days."""
    from vlstm import predict
    
    if X_full is None or len(X_full) < lookback:
        return {"signal": "N/A", "probabilities": {}, "confidence": 0, "top_features": []}
    
    X_recent = X_full[-lookback:]
    X_scaled = (X_recent - scale_mean) / scale_std
    X_seq = X_scaled.reshape(1, lookback, -1)
    
    preds, probs, attn = predict(model, X_seq)
    
    signal_idx = preds[0]
    signal = target_etfs[signal_idx]
    probabilities = {target_etfs[i]: float(probs[0, i]) for i in range(len(target_etfs))}
    confidence = float(probs[0, signal_idx])
    
    return {
        "signal": signal,
        "probabilities": probabilities,
        "confidence": confidence,
        "top_features": [],
    }


def summarise_window_result(window, train_res, backtest, live_signal, lookback, option):
    """
    Summarise all results for one window into a dict.
    CRITICAL: Always store live_signal as a proper dict.
    """
    # CRITICAL FIX: Force live_signal to be a dict
    if isinstance(live_signal, dict):
        live_signal_dict = live_signal
    else:
        # If it's a string or something else, convert to dict
        live_signal_dict = {
            "signal": str(live_signal) if live_signal else "N/A",
            "confidence": 0,
            "probabilities": {},
            "top_features": []
        }
    
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
        # CRITICAL: Always a dict
        "live_signal": live_signal_dict,
    }


def stream_consensus(results: List[Dict], target_etfs: List[str]) -> Dict[str, Any]:
    """Compute consensus across all windows in a stream."""
    if not results:
        return {
            "signal": "N/A",
            "strength": "none",
            "score_pct": 0,
            "agreement": 0,
            "votes": {},
            "z_score": 0,
        }
    
    votes = {etf: 0 for etf in target_etfs}
    for r in results:
        live = r.get("live_signal", {})
        if isinstance(live, dict):
            signal = live.get("signal", "N/A")
        else:
            signal = str(live) if live else "N/A"
        if signal in votes:
            votes[signal] += 1
    
    total = len(results)
    winner = max(votes, key=votes.get)
    winner_count = votes[winner]
    
    agreement = (winner_count / total) * 100 if total > 0 else 0
    
    vote_counts = list(votes.values())
    mean_votes = np.mean(vote_counts)
    std_votes = np.std(vote_counts)
    z_score = (winner_count - mean_votes) / std_votes if std_votes > 0 else 0
    
    if agreement >= 60 and z_score > 1.5:
        strength = "high"
    elif agreement >= 40:
        strength = "moderate"
    else:
        strength = "low"
    
    score_pct = agreement
    
    return {
        "signal": winner,
        "strength": strength,
        "score_pct": round(score_pct, 1),
        "agreement": round(agreement, 1),
        "votes": votes,
        "z_score": round(float(z_score), 2),
        "total_windows": total,
    }
