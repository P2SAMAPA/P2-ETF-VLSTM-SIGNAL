"""
conviction.py
P2-ETF-VLSTM-SIGNAL

Conviction scoring: converts raw softmax probabilities into a
Z-score conviction measure and human-readable label.

Also summarises VSN attention weights for display.
"""

import numpy as np


# ── Z-score conviction ────────────────────────────────────────────────────────

def compute_conviction(
    proba:       np.ndarray,    # [n_etfs] softmax probabilities (single row)
    target_etfs: list,
) -> dict:
    """
    Z-score of the top ETF's probability vs the distribution of all ETF probabilities.
    Z = (p_top - mean(p)) / std(p)
    Higher Z = more decisive model = higher conviction.
    """
    if len(proba) == 0:
        return {"z_score": 0.0, "label": "None", "signal": None, "proba": {}}

    top_idx = int(np.argmax(proba))
    p_top   = float(proba[top_idx])
    mean_p  = float(np.mean(proba))
    std_p   = float(np.std(proba)) + 1e-8
    z_score = (p_top - mean_p) / std_p

    signal  = target_etfs[top_idx] if top_idx < len(target_etfs) else "?"

    proba_dict = {
        target_etfs[i]: round(float(proba[i]), 4)
        for i in range(min(len(target_etfs), len(proba)))
    }

    return {
        "z_score":  round(float(z_score), 3),
        "label":    _conviction_label(z_score),
        "signal":   signal,
        "proba":    proba_dict,
        "p_top":    round(p_top, 4),
    }


def _conviction_label(z: float) -> str:
    if z >= 2.5:   return "Very High"
    elif z >= 1.5: return "High"
    elif z >= 0.8: return "Moderate"
    elif z >= 0.0: return "Low"
    else:          return "Weak"


# ── VSN attention summary ─────────────────────────────────────────────────────

def summarise_vsn_attention(
    attn:          np.ndarray,
    feature_names: list,
    top_k:         int = 10,
) -> list:
    """
    Average VSN attention over time and lookback.
    Returns top-k features sorted by mean attention weight.
    """
    if attn.ndim == 3:
        mean_attn = attn.mean(axis=(0, 1))
    elif attn.ndim == 2:
        mean_attn = attn.mean(axis=0)
    else:
        mean_attn = attn

    n   = min(len(feature_names), len(mean_attn))
    idx = np.argsort(mean_attn[:n])[::-1][:top_k]

    return [
        {"feature": feature_names[i], "weight": round(float(mean_attn[i]), 5), "rank": rank + 1}
        for rank, i in enumerate(idx)
    ]


def group_vsn_attention(attn_summary: list) -> dict:
    """Group top VSN features by type: Returns / MACD / Volatility / Macro / Other."""
    groups = {"Returns": [], "MACD": [], "Volatility": [], "Macro": [], "Other": []}
    macro_keys = {"VIX", "DXY", "T10Y2Y", "TBILL_3M", "IG_SPREAD", "HY_SPREAD"}

    for item in attn_summary:
        f = item["feature"]
        if any(f.endswith(f"_r{h}d") for h in [1, 5, 21, 63, 126, 252]):
            groups["Returns"].append(item)
        elif "macd" in f:
            groups["MACD"].append(item)
        elif "_vol" in f or "_vsfactor" in f:
            groups["Volatility"].append(item)
        elif any(m in f for m in macro_keys):
            groups["Macro"].append(item)
        else:
            groups["Other"].append(item)

    return {k: v for k, v in groups.items() if v}


# ── Window-level conviction aggregator ───────────────────────────────────────

def window_conviction_summary(results: list) -> list:
    """Conviction summary for each window's live signal."""
    summary = []
    for r in results:
        if not r or not r.get("live_signal"):
            continue
        proba = np.array(list(r["live_signal"]["proba"].values()))
        etfs  = list(r["live_signal"]["proba"].keys())
        conv  = compute_conviction(proba, etfs)
        summary.append({
            "label":      r.get("label", ""),
            "stream":     r.get("stream", ""),
            "option":     r.get("option", ""),
            "loss_mode":  r.get("loss_mode", ""),
            "signal":     conv["signal"],
            "z_score":    conv["z_score"],
            "conviction": conv["label"],
            "proba":      conv["proba"],
        })
    return summary
