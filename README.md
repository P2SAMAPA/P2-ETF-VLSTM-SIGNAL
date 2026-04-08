# P2-ETF-VLSTM-SIGNAL

Daily ETF signal generator using a **Variable Selection Network + LSTM** (VLSTM) architecture. Trained across two streams of rolling windows to test how much history matters and which regime the market is currently in.

**Now supports two ETF universes:**
- **FI (Fixed Income)**: TLT, VNQ, SLV, GLD, HYG, LQD
- **Equity**: GDX, XLV, XLY, XLK, XLE, XLI

---

## Research Foundation

This project implements and adapts the VLSTM architecture from:

> **Deep Learning for Financial Time Series: A Large-Scale Benchmark of Risk-Adjusted Performance**  
> Adir Saly-Kaufmann, Kieran Wood, Jan Peter-Calliess, Stefan Zohren  
> *Machine Learning Research Group, Department of Engineering Science, University of Oxford*  
> *Oxford-Man Institute of Quantitative Finance, University of Oxford*

In the paper's benchmark (Table 2), VLSTM achieves the highest raw annualised return (23.9%) and CAGR (26.3%) across all evaluated architectures, outperforming LPatchTST, PatchTST, and other deep learning baselines on financial time series.

---

## What It Does

Every weekday after market close, GitHub Actions runs two parallel workflows:

1. **FI Universe** (21:30 UTC / 4:30 PM EST)
   - Trains expanding and shrinking streams **in parallel**
   - Outputs: `fi_expanding_latest.json`, `fi_shrinking_latest.json`, `fi_history.parquet`

2. **Equity Universe** (22:00 UTC / 5:00 PM EST)
   - Trains expanding and shrinking streams **in parallel**
   - Outputs: `equity_expanding_latest.json`, `equity_shrinking_latest.json`, `equity_history.parquet`

Each workflow:
- Pulls the latest price/macro data from HuggingFace
- Trains 4 models per window × ~30 windows per stream (~120 training runs total)
- Each model produces a **live next-day signal** for the 6 target ETFs
- Results are pushed to a HuggingFace output dataset
- Streamlit reads those results and displays consensus signals + per-window breakdowns

The Streamlit app is **display-only** — no training happens in the UI.

---

## ETF Universes

### FI Universe (Fixed Income + Commodities)

| ETF | Asset Class |
|-----|-------------|
| TLT | Long-term US Treasuries |
| VNQ | US Real Estate (REITs) |
| SLV | Silver |
| GLD | Gold |
| HYG | High Yield Corporate Bonds |
| LQD | Investment Grade Corporate Bonds |

**Benchmarks (features only)**: SPY, AGG  
**Macro features (Option A)**: VIX, DXY, T10Y2Y, TBILL_3M, IG_SPREAD, HY_SPREAD

### Equity Universe (Sectors + Gold Miners)

| ETF | Asset Class |
|-----|-------------|
| GDX | Gold Miners |
| XLV | Health Care |
| XLY | Consumer Discretionary |
| XLK | Technology |
| XLE | Energy |
| XLI | Industrials |

**Benchmarks (features only)**: SPY, QQQ

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

Close-derived features per ticker (8 tickers: 6 ETFs + 2 benchmarks):

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
2008→2010  →  predicts 2026-04-08
2008→2011  →  predicts 2026-04-08
...
2008→2025  →  predicts 2026-04-08
```

### Shrinking Stream

Progressively less distant history — tests how far back matters:

```
2008→2025  →  predicts 2026-04-08
2009→2025  →  predicts 2026-04-08
...
2024→2025  →  predicts 2026-04-08
```

Every window — regardless of training period — uses the **same trained weights** applied to the most recent `lookback` rows of the full dataset. Each window's scaler is fitted only on its own training data, ensuring no lookahead contamination.

This enables true like-for-like comparison: does the 2008→2010 model (no post-GFC memory) agree with 2008→2025 on tomorrow's signal? Disagreement reveals regime instability.

---

## File Structure

```
P2-ETF-VLSTM-SIGNAL/
├── .github/
│   └── workflows/
│       ├── retrain_fi_parallel.yml      # FI workflow (parallel streams)
│       └── retrain_equity_parallel.yml  # Equity workflow (parallel streams)
├── config.py              # Universe configurations (FI + Equity)
├── loader.py              # Data loading + feature engineering
├── vlstm.py               # VLSTM model, training loop, both loss modes
├── backtest.py            # Backtest engine + live signal generator
├── conviction.py          # Z-score conviction + VSN attention summary
├── writer.py              # HuggingFace dataset writer + reader (fixed schema)
├── train.py               # Main orchestrator (GitHub Actions entry point)
├── app.py                 # Streamlit display UI
├── cleanup_hf_dataset.py  # Utility to clean up corrupted HF dataset
└── requirements.txt
```

---

## HuggingFace Datasets

### Input Datasets (per universe)

| Dataset | Universe | Purpose |
|---------|----------|---------|
| `P2SAMAPA/fi-etf-macro-signal-master-data` | FI | Input: daily OHLCV + macro data |
| `P2SAMAPA/equity-etf-macro-signal-master-data` | Equity | Input: daily OHLCV + macro data |

### Output Dataset (shared, prefixed by universe)

| Dataset | Purpose |
|---------|---------|
| `P2SAMAPA/p2-etf-vlstm-outputs` | Output: signals, metrics, window results |

Output files (prefixed by universe):

| File | Description |
|------|-------------|
| `fi_expanding_latest.json` | FI expanding stream full results (overwritten daily) |
| `fi_shrinking_latest.json` | FI shrinking stream full results (overwritten daily) |
| `fi_history.parquet` | FI consensus history (one row per run date) |
| `equity_expanding_latest.json` | Equity expanding stream full results (overwritten daily) |
| `equity_shrinking_latest.json` | Equity shrinking stream full results (overwritten daily) |
| `equity_history.parquet` | Equity consensus history (one row per run date) |

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

## GitHub Actions

### Parallel Workflows (Recommended)

| Workflow | Schedule | Streams | Time |
|----------|----------|---------|------|
| `retrain_fi_parallel.yml` | 21:30 UTC weekdays | Expanding + Shrinking (parallel) | ~45 min |
| `retrain_equity_parallel.yml` | 22:00 UTC weekdays | Expanding + Shrinking (parallel) | ~45 min |

**Total daily training time: ~90 minutes** (50% faster than sequential)

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
python train.py --universe fi --benchmark
python train.py --universe equity --benchmark

# Full training run - FI
python train.py --universe fi --stream expanding
python train.py --universe fi --stream shrinking

# Full training run - Equity
python train.py --universe equity --stream expanding
python train.py --universe equity --stream shrinking

# Streamlit UI
streamlit run app.py
```

---

## Output Schema (Fixed)

### Consensus Structure

```json
{
  "stream": "expanding",
  "run_date": "2026-04-08",
  "data_through": "2026-04-07",
  "consensus": {
    "signal": "GDX",
    "strength": "high",
    "score_pct": 75.0,
    "agreement": 60.0,
    "votes": {
      "GDX": 5,
      "XLV": 3,
      "XLY": 2,
      "XLK": 1,
      "XLE": 0,
      "XLI": 0
    },
    "z_score": 2.1,
    "total_windows": 30
  },
  "universe": "equity",
  "target_etfs": ["GDX", "XLV", "XLY", "XLK", "XLE", "XLI"],
  "windows": [...]
}
```

**Note:** ETF votes are nested in the `votes` dict to prevent schema conflicts between FI and Equity universes.

### History Parquet Schema

| Column | Description |
|--------|-------------|
| `run_date` | Date of the run (YYYY-MM-DD) |
| `data_through` | Last date of training data |
| `exp_signal` | Expanding stream consensus signal |
| `exp_strength` | Expanding stream strength (high/moderate/low) |
| `exp_score_pct` | Expanding stream agreement percentage |
| `exp_agreement` | Expanding stream vote agreement |
| `shr_signal` | Shrinking stream consensus signal |
| `shr_strength` | Shrinking stream strength |
| `shr_score_pct` | Shrinking stream agreement percentage |
| `shr_agreement` | Shrinking stream vote agreement |
| `universe` | "fi" or "equity" |

---

## Interpreting the Output

### Signal Strength

| Signal strength | Meaning |
|-----------------|---------|
| Strong Consensus (≥60% windows agree) | Clear directional call |
| Majority Signal (40–59%) | Moderate agreement |
| Split Signal (<40%) | Regime uncertainty — disagreement across windows |

When the expanding and shrinking streams agree → higher confidence.

When they disagree → the market may be in a transitional regime where historical patterns are less reliable.

### VSN Attention Panel

The VSN attention panel reveals **why** the model made its call: on days where VIX or T10Y2Y has high gate weight, macro risk/yield regime is driving the signal; on days dominated by MACD features, momentum is the primary driver.

---

## Deployment Checklist

1. **Create HF input datasets** (if not already done):
   - `P2SAMAPA/fi-etf-macro-signal-master-data`
   - `P2SAMAPA/equity-etf-macro-signal-master-data`

2. **Run cleanup script** (if migrating from old single-universe setup):
   ```bash
   export HF_TOKEN=your_token
   python cleanup_hf_dataset.py
   ```

3. **Set GitHub secret**:
   - `HF_TOKEN` — HuggingFace write token

4. **Update Streamlit app** to handle new file prefixes:
   ```python
   # Load FI data
   fi_data = load_latest(hf_token, file_prefix="fi")

   # Load Equity data
   equity_data = load_latest(hf_token, file_prefix="equity")
   ```

5. **Test workflows manually** before enabling schedules

---

## Troubleshooting

### Schema Casting Errors
If you see errors like `Couldn't cast array of type struct<GDX: int64...>`, this means:
- Old mixed-schema data exists in the dataset
- Run `cleanup_hf_dataset.py` to fix

### Silent Failures (Green check but no upload)
The corrected workflows now:
- Exit with code 1 on upload failure (red X in Actions)
- Include verification steps to confirm files exist

### Timeout Issues
If workflows timeout:
- Reduce `epochs` in workflow_dispatch
- Switch from parallel to sequential workflows
- Run benchmark to check timing estimates

---

## License

MIT License — feel free to adapt for your own use. Attribution to the original Oxford paper is appreciated.
