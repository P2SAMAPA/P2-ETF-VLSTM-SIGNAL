# P2-ETF-VLSTM-SIGNAL

Daily ETF signal generator using a **Variable Selection Network + LSTM** (VLSTM) architecture. Trained across two streams of rolling windows to test how much history matters and which regime the market is currently in.

---

## Research Foundation

This project implements and adapts the VLSTM architecture from:

> **Deep Learning for Financial Time Series: A Large-Scale Benchmark of Risk-Adjusted Performance**
> Adir Saly-Kaufmann, Kieran Wood, Jan Peter-Calliess, Stefan Zohren
> *Machine Learning Research Group, Department of Engineering Science, University of Oxford*
> *Oxford-Man Institute of Quantitative Finance, University of Oxford*

In the paper's benchmark (Table 2), VLSTM achieves the highest raw annualised return (23.9%) and CAGR (26.3%) across all evaluated architectures, outperforming LPatchTST, PatchTST, and other deep learning baselines on financial time series.

**Key adaptations in this project vs the paper:**

| Aspect | Paper | This Project |
|--------|-------|-------------|
| Task | Portfolio position sizing (tanh output) | ETF selection (softmax over 6 ETFs) |
| Loss | Sharpe ratio (end-to-end) | Both CE and Sharpe — validation picks winner |
| Asset universe | Large cross-section | 6 ETFs + SPY/AGG benchmarks |
| Ticker embeddings | Yes | No — universe too small |
| Macro features | Close-derived only | Option A adds VIX, DXY, T10Y2Y, spreads |
| Feature engineering | Paper's exact spec | Paper's spec faithfully replicated in `loader.py` |

---

## What It Does

Every weekday after market close, GitHub Actions:

1. Pulls the latest price/macro data from HuggingFace
2. Trains 4 models per window × ~30 windows across two streams (~120 training runs)
3. Each model produces a **live next-day signal** for the 6 target ETFs
4. Results are pushed to a HuggingFace output dataset
5. Streamlit reads those results and displays consensus signals + per-window breakdowns

The Streamlit app is **display-only** — no training happens in the UI.

---

## ETF Universe

| ETF | Asset Class |
|-----|-------------|
| TLT | Long-term US Treasuries |
| VNQ | US Real Estate (REITs) |
| SLV | Silver |
| GLD | Gold |
| HYG | High Yield Corporate Bonds |
| LQD | Investment Grade Corporate Bonds |

**Benchmarks (features only):** SPY, AGG  
**Macro features (Option A only):** VIX, DXY, T10Y2Y, TBILL_3M, IG_SPREAD, HY_SPREAD

---

## Architecture

### VLSTM (Variable Selection Network + LSTM)

```
Input [batch, lookback, n_features]
        ↓
VSN — per-feature independent embedding
    → gating network → softmax weights
    → weighted sum → [batch, lookback, hidden_dim=128]
        ↓
LSTM — 2 layers, hidden_dim=128
    → final hidden state h_T
        ↓
LayerNorm → Dropout → Linear → [batch, n_etfs]
        ↓
Softmax → ETF probabilities
```

The VSN produces **explicit attention weights** per feature per timestep — logged as top-10 most attended features, enabling interpretability of what the model is focusing on each day.

### Two Feature Options

| Option | Features | Count |
|--------|----------|-------|
| A | Close-derived + macro | ~94 |
| B | Close-derived only | 88 |

Close-derived features per ticker (8 tickers: 6 ETFs + SPY + AGG):
- Vol-normalised returns: 1d, 5d, 21d, 63d, 126d, 252d
- MACD signals: (4,12), (8,24), (32,96) — double-normalised
- EWMA volatility (span=63), normalised by 252d rolling mean
- Vol-scaling factor 1/σ clipped at 400

### Two Loss Modes

| Mode | Loss | Description |
|------|------|-------------|
| `ce` | Cross-entropy | Argmax label: which ETF had best next-day return |
| `sharpe` | Differentiable Sharpe | Softmax weights × next-day returns → −annualised Sharpe |

Validation Sharpe selects the winner across all 4 models per window.

---

## Window Streams

### Expanding Stream
Progressively more history — tests whether old regimes (e.g. GFC) still add value:

```
2008→2010  →  predicts 2026-03-13
2008→2011  →  predicts 2026-03-13
...
2008→2025  →  predicts 2026-03-13
```

### Shrinking Stream
Progressively less distant history — tests how far back matters:

```
2008→2025  →  predicts 2026-03-13
2009→2025  →  predicts 2026-03-13
...
2024→2025  →  predicts 2026-03-13
```

Every window — regardless of training period — uses the **same trained weights** applied to the most recent `lookback` rows of the full dataset. Each window's scaler is fitted only on its own training data, ensuring no lookahead contamination.

This enables true like-for-like comparison: does the 2008→2010 model (no post-GFC memory) agree with 2008→2025 on tomorrow's signal? Disagreement reveals regime instability.

---

## File Structure

```
P2-ETF-VLSTM-SIGNAL/
├── .github/
│   └── workflows/
│       └── daily_retrain.yml   ← GitHub Actions: trains daily at 4:30pm EST
├── loader.py                   ← Data loading + feature engineering
├── vlstm.py                    ← VLSTM model, training loop, both loss modes
├── backtest.py                 ← Backtest engine + live signal generator
├── conviction.py               ← Z-score conviction + VSN attention summary
├── writer.py                   ← HuggingFace dataset writer + reader
├── train.py                    ← Main orchestrator (GitHub Actions entry point)
├── app.py                      ← Streamlit display UI
└── requirements.txt
```

---

## HuggingFace Datasets

| Dataset | Purpose |
|---------|---------|
| `P2SAMAPA/fi-etf-macro-signal-master-data` | Input: daily OHLCV + macro data |
| `P2SAMAPA/p2-etf-vlstm-outputs` | Output: signals, metrics, window results |

Output dataset files:
- `latest.json` — full results from most recent run (overwritten daily)
- `history.parquet` — one row per run date, consensus signals only

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Hidden dim | 128 |
| LSTM layers | 2 |
| Dropout | 0.3 |
| Learning rate | 5e-4 |
| Weight decay | 1e-5 |
| Batch size | 64 |
| Max epochs | 80 |
| Early stopping patience | 15 |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=7) |
| Gradient clip norm | 1.0 |
| Lookback candidates | 30d, 45d, 60d (auto-selected on val Sharpe) |
| Train/Val/Test split | 70% / 15% / 15% chronological |
| Transaction cost | 10 bps on turnover |

---

## Streamlit UI

Three tabs:

**📈 Expanding Stream**
- Consensus banner: winning ETF, vote count, agreement %
- Signal distribution chart across all windows
- Per-window expanders (scrollable) showing:
  - Live next-day signal + probability bar chart
  - Backtest metrics: ann. return, Sharpe, max drawdown, hit rate
  - VSN top-10 attention features (colour-coded by type)
  - All 4 models comparison table
  - Last 30 test-set predictions audit trail

**📉 Shrinking Stream** — same structure

**🕐 History** — daily consensus signal history, both streams

---

## GitHub Actions

**Trigger:** `cron: '30 21 * * 1-5'` (21:30 UTC = 4:30pm EST, weekdays)

**Manual trigger options:**
- `benchmark=true` — times a single window × 4 models and extrapolates full runtime
- `epochs=N` — override max epochs (default 80)

**Required secret:** `HF_TOKEN` — HuggingFace token with write access to `P2SAMAPA/p2-etf-vlstm-outputs`

### Benchmark Mode

Before committing to daily full retraining, run the benchmark first:

```
workflow_dispatch → benchmark: true
```

This trains one window (2008→2020) with all 4 models, prints per-model timing, and extrapolates to the full ~30-window run. Use this to decide whether to retrain daily, weekly, or on a hybrid schedule.

---

## Running Locally

```bash
pip install -r requirements.txt

# Set HF token
export HF_TOKEN=hf_...

# Benchmark timing only
python train.py --benchmark

# Full training run
python train.py

# Streamlit UI
streamlit run app.py
```

---

## Interpreting the Output

| Signal strength | Meaning |
|----------------|---------|
| Strong Consensus (≥60% windows agree) | Clear directional call |
| Majority Signal (40–59%) | Moderate agreement |
| Split Signal (<40%) | Regime uncertainty — disagreement across windows |

When the expanding and shrinking streams agree → higher confidence.  
When they disagree → the market may be in a transitional regime where historical patterns are less reliable.

The VSN attention panel reveals **why** the model made its call: on days where VIX or T10Y2Y has high gate weight, macro risk/yield regime is driving the signal; on days dominated by MACD features, momentum is the primary driver.
