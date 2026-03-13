"""
backtest.py
P2-ETF-VLSTM-SIGNAL

Backtest engine: given model predictions on the test set, simulate
a daily strategy that holds one ETF per day (the model's top pick).

Focus is on raw annualised return (not risk-adjusted) as the primary
metric, consistent with the project goal of maximising returns.

Also computes:
  - Sharpe ratio
  - Max drawdown
  - Hit rate (correctly predicted best ETF)
  - Equity curve (daily cumulative returns)
  - Audit trail (last 30 days)

Key design: every window (expanding and shrinking) runs both:
  1. Backtest on its own test slice  → historical performance metrics
  2. Live prediction on latest data  → next-day signal for 2026-03-13
"""

import numpy as np
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────

TRADING_DAYS = 252


# ── Core strategy executor ────────────────────────────────────────────────────

def execute_strategy(
    preds:       np.ndarray,        # [T] predicted ETF index
    proba:       np.ndarray,        # [T, n_etfs] softmax probabilities
    y_returns:   np.ndarray,        # [T, n_etfs] actual next-day returns
    dates:       pd.DatetimeIndex,
    target_etfs: list,
    fee_bps:     float = 10.0,
) -> dict:
    """
    Simulate daily single-ETF hold strategy on the test set.

    At each step t:
      - Hold ETF with highest model probability
      - Earn that ETF's actual next-day return
      - Pay fee_bps on turnover (when ETF changes)

    Returns full performance metrics + equity curve + audit trail.
    """
    n      = len(preds)
    n_etfs = len(target_etfs)
    fee    = fee_bps / 10_000.0

    daily_rets = np.zeros(n)
    etf_held   = []
    prev_pred  = -1

    for t in range(n):
        pred = int(preds[t])
        ret  = float(y_returns[t, pred]) if pred < n_etfs else 0.0

        if prev_pred != pred and prev_pred != -1:
            ret -= fee

        daily_rets[t] = ret
        etf_held.append(target_etfs[pred] if pred < n_etfs else "CASH")
        prev_pred = pred

    ann_return = _annualised_return(daily_rets)
    sharpe     = _sharpe_ratio(daily_rets)
    max_dd     = _max_drawdown(daily_rets)
    hit_rate   = _hit_rate(preds, y_returns)

    equity = pd.Series(
        np.cumprod(1 + daily_rets),
        index=dates,
        name="strategy",
    )

    # Audit trail — last 30 days
    audit = []
    for t in range(max(0, n - 30), n):
        pred   = int(preds[t])
        actual = int(np.argmax(y_returns[t]))
        audit.append({
            "date":        str(dates[t].date()),
            "predicted":   target_etfs[pred] if pred < n_etfs else "?",
            "actual_best": target_etfs[actual] if actual < n_etfs else "?",
            "correct":     pred == actual,
            "return_pct":  round(daily_rets[t] * 100, 3),
            "proba": {
                target_etfs[i]: round(float(proba[t, i]), 4)
                for i in range(min(n_etfs, proba.shape[1]))
            },
        })

    # Next signal from test set (not live — use live_prediction() for that)
    last_pred   = int(preds[-1])
    next_signal = target_etfs[last_pred] if last_pred < n_etfs else "CASH"

    return {
        "ann_return":    float(ann_return),
        "sharpe":        float(sharpe),
        "max_dd":        float(max_dd),
        "hit_rate":      float(hit_rate),
        "n_days":        n,
        "equity":        equity,
        "daily_rets":    daily_rets,
        "etf_held":      etf_held,
        "audit_trail":   audit,
        "etf_breakdown": etf_breakdown(preds, y_returns, target_etfs),
        "next_signal":   next_signal,
        "dates":         dates,
    }


# ── Live next-day prediction ──────────────────────────────────────────────────

def live_prediction(
    proba:       np.ndarray,    # [n_etfs] softmax probabilities for latest row
    target_etfs: list,
) -> dict:
    """
    Extract the live next-day signal from the latest probability vector.

    Every window — regardless of training period — applies its learned
    weights to the most recent lookback window (up to 2026-03-12) to
    forecast 2026-03-13. This enables true like-for-like comparison
    across all windows and both streams.
    """
    pred       = int(np.argmax(proba))
    signal     = target_etfs[pred]
    proba_dict = {
        target_etfs[i]: round(float(proba[i]), 4)
        for i in range(len(target_etfs))
    }
    return {
        "signal":     signal,
        "proba":      proba_dict,
        "confidence": round(float(proba[pred]), 4),
    }


# ── Per-ETF breakdown ─────────────────────────────────────────────────────────

def etf_breakdown(
    preds:       np.ndarray,
    y_returns:   np.ndarray,
    target_etfs: list,
) -> list:
    """How often each ETF was selected and its average return when selected."""
    rows = []
    for i, etf in enumerate(target_etfs):
        mask       = preds == i
        n_selected = int(mask.sum())
        if n_selected == 0:
            rows.append({
                "etf": etf, "n_selected": 0,
                "pct_of_days": 0.0, "avg_return_pct": 0.0,
            })
            continue
        avg_ret = float(y_returns[mask, i].mean() * 100)
        rows.append({
            "etf":            etf,
            "n_selected":     n_selected,
            "pct_of_days":    round(n_selected / len(preds) * 100, 1),
            "avg_return_pct": round(avg_ret, 3),
        })
    return sorted(rows, key=lambda r: -r["n_selected"])


# ── SPY benchmark ─────────────────────────────────────────────────────────────

def spy_benchmark(df_raw: pd.DataFrame, dates: pd.DatetimeIndex) -> dict:
    """SPY buy-and-hold over the same test date range."""
    if "SPY" not in df_raw.columns:
        return {"ann_return": None, "equity": None}

    spy_rets   = df_raw["SPY"].reindex(dates).ffill().pct_change().fillna(0).values
    spy_equity = pd.Series(np.cumprod(1 + spy_rets), index=dates, name="SPY")

    return {
        "ann_return": float(_annualised_return(spy_rets)),
        "equity":     spy_equity,
    }


# ── Metric helpers ────────────────────────────────────────────────────────────

def _annualised_return(daily_rets: np.ndarray) -> float:
    if len(daily_rets) == 0:
        return 0.0
    cum = float(np.prod(1.0 + daily_rets))
    n   = len(daily_rets)
    return float(cum ** (TRADING_DAYS / n) - 1)


def _sharpe_ratio(daily_rets: np.ndarray) -> float:
    if len(daily_rets) < 5:
        return 0.0
    mu  = float(np.mean(daily_rets))
    std = float(np.std(daily_rets)) + 1e-8
    return float(mu / std * np.sqrt(TRADING_DAYS))


def _max_drawdown(daily_rets: np.ndarray) -> float:
    if len(daily_rets) == 0:
        return 0.0
    cum  = np.cumprod(1.0 + daily_rets)
    peak = np.maximum.accumulate(cum)
    dd   = (cum - peak) / (peak + 1e-8)
    return float(dd.min())


def _hit_rate(preds: np.ndarray, y_returns: np.ndarray) -> float:
    if len(preds) == 0:
        return 0.0
    actual_best = np.argmax(y_returns, axis=1)
    return float((preds == actual_best).mean())


# ── Window comparison table ───────────────────────────────────────────────────

def build_window_comparison(results: list) -> pd.DataFrame:
    """
    Build a summary DataFrame from a list of window result dicts.
    One row per trained model (window × option × loss_mode).
    """
    rows = []
    for r in results:
        if r is None:
            continue
        live = r.get("live_signal", {}) or {}
        rows.append({
            "Window":      r.get("label", ""),
            "Stream":      r.get("stream", ""),
            "Option":      r.get("option", ""),
            "Loss":        r.get("loss_mode", ""),
            "Live Signal": live.get("signal", "—"),
            "Confidence":  f"{live.get('confidence', 0)*100:.1f}%",
            "Ann. Return": f"{r.get('ann_return', 0)*100:.2f}%",
            "Sharpe":      f"{r.get('sharpe', 0):.2f}",
            "Max DD":      f"{r.get('max_dd', 0)*100:.2f}%",
            "Hit Rate":    f"{r.get('hit_rate', 0)*100:.1f}%",
            "Train Time":  f"{r.get('train_time_s', 0):.1f}s",
        })
    return pd.DataFrame(rows)


# ── Live signal generator (called from train.py) ──────────────────────────────

def generate_live_signal(
    model,
    X_full:      np.ndarray,    # full-dataset features (unscaled), shape [T, n_features]
    scale_mean:  np.ndarray,    # scaler fitted on this window's train set
    scale_std:   np.ndarray,
    lookback:    int,
    target_etfs: list,
    feature_names: list,
) -> dict:
    """
    Apply a window's trained model to the most recent `lookback` rows of the
    full dataset to generate the live next-day signal.

    Scaling uses this window's own scaler — critical for like-for-like comparison.
    The model never saw this data during training.
    """
    from vlstm import predict, build_sequences

    if X_full is None or len(X_full) < lookback:
        return {"signal": None, "proba": {}, "confidence": 0.0,
                "vsn_top_features": [], "error": "insufficient data"}

    # Take last lookback+1 rows → build one sequence of length lookback
    window_data = X_full[-(lookback + 1):]

    # Scale using this window's training statistics
    X_scaled = (window_data - scale_mean) / (scale_std + 1e-8)
    X_scaled = np.clip(X_scaled, -10, 10)

    # Build single sequence [1, lookback, n_features]
    X_seq = X_scaled[np.newaxis, :-1, :]     # shape [1, lookback, n_features]

    try:
        preds, proba, attn = predict(model, X_seq)
    except Exception as e:
        return {"signal": None, "proba": {}, "confidence": 0.0,
                "vsn_top_features": [], "error": str(e)}

    pred       = int(preds[0])
    proba_vec  = proba[0]       # [n_etfs]
    signal     = target_etfs[pred] if pred < len(target_etfs) else "?"
    confidence = float(proba_vec[pred])

    proba_dict = {
        target_etfs[i]: round(float(proba_vec[i]), 4)
        for i in range(min(len(target_etfs), len(proba_vec)))
    }

    # VSN attention for this live slice — average over the lookback window
    vsn_top = []
    if attn is not None and len(feature_names) > 0:
        from conviction import summarise_vsn_attention
        vsn_top = summarise_vsn_attention(attn, feature_names, top_k=10)

    return {
        "signal":         signal,
        "proba":          proba_dict,
        "confidence":     round(confidence, 4),
        "vsn_top_features": vsn_top,
    }


# ── Window result assembler ───────────────────────────────────────────────────

def summarise_window_result(
    window:     dict,
    train_res:  dict,
    bt:         dict,
    live:       dict,
    lookback:   int,
    option:     str,
) -> dict:
    """
    Assemble a flat result dict for one trained model (window × option × loss_mode).
    This is what gets stored in HF and displayed in the UI.

    Includes:
      - Window metadata (label, stream, dates)
      - Training metadata (val_sharpe, epochs, loss_mode)
      - Backtest metrics on test set (historical performance)
      - Live next-day signal for this window's model
      - Equity curve kept for UI charting
    """
    loss_mode = train_res.get("loss_mode", "?")
    return {
        # Identity
        "label":      window.get("label", ""),
        "stream":     window.get("stream", ""),
        "start_year": window.get("start_year"),
        "end_year":   window.get("end_year"),
        "option":     option,
        "loss_mode":  loss_mode,
        "lookback":   lookback,

        # Training quality
        "val_sharpe":   float(train_res.get("val_sharpe", 0.0)),
        "train_loss":   float(train_res.get("train_loss", 0.0)),
        "epochs_run":   int(train_res.get("epochs_run", 0)),

        # Backtest metrics (test set — historical window performance)
        "ann_return":   float(bt.get("ann_return", 0.0)),
        "sharpe":       float(bt.get("sharpe", 0.0)),
        "max_dd":       float(bt.get("max_dd", 0.0)),
        "hit_rate":     float(bt.get("hit_rate", 0.0)),
        "n_days":       int(bt.get("n_days", 0)),
        "equity":       bt.get("equity"),           # pd.Series — stripped by writer
        "audit_trail":   bt.get("audit_trail", []),
        "etf_breakdown": bt.get("etf_breakdown", []),

        # Live signal — every window predicts next trading day
        "live_signal":  live,
        "live_date":    "2026-03-13",    # today + 1 trading day
    }


# ── Stream consensus ──────────────────────────────────────────────────────────

def stream_consensus(results: list, target_etfs: list) -> dict:
    """
    Aggregate live signals across all windows in a stream using vote count.
    Average confidence used as tiebreaker.

    Returns consensus signal, vote breakdown, and strength label.
    """
    from collections import Counter, defaultdict

    vote_counts   = Counter()
    conf_sum      = defaultdict(float)
    valid         = [r for r in results if r and r.get("live_signal")]

    if not valid:
        return {"signal": None, "vote_counts": {}, "strength": "No data",
                "score_pct": 0, "total_windows": 0}

    for r in valid:
        sig  = r["live_signal"]["signal"]
        conf = r["live_signal"]["confidence"]
        vote_counts[sig] += 1
        conf_sum[sig]    += conf

    total      = sum(vote_counts.values())
    top_signal = vote_counts.most_common(1)[0][0]
    top_votes  = vote_counts[top_signal]
    score_pct  = top_votes / total * 100

    avg_conf = {
        etf: round(conf_sum[etf] / vote_counts[etf], 4)
        for etf in vote_counts
    }

    if score_pct >= 60:
        strength = "Strong Consensus"
    elif score_pct >= 40:
        strength = "Majority Signal"
    else:
        strength = "Split Signal"

    return {
        "signal":         top_signal,
        "vote_counts":    dict(vote_counts.most_common()),
        "avg_confidence": avg_conf,
        "score_pct":      round(score_pct, 1),
        "total_windows":  total,
        "top_votes":      top_votes,
        "strength":       strength,
    }
