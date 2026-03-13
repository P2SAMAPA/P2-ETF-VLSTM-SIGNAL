"""
train.py
P2-ETF-VLSTM-SIGNAL

Main training script — runs in GitHub Actions daily after market close.

Flow:
  1. Load data from HF
  2. For each window in expanding + shrinking streams:
     For each option (A, B) × loss_mode (ce, sharpe) → 4 models per window:
       a. Build features
       b. Find best lookback (30/45/60d)
       c. Train VLSTM
       d. Backtest on test slice
       e. Generate live next-day signal (2026-03-13)
       f. Pick best of 4 models by val_sharpe
  3. Compute stream consensus (expanding / shrinking)
  4. Write outputs to HF dataset

Benchmark mode (--benchmark):
  Trains a single window with all 4 models, reports timing,
  extrapolates to full run. Use this first to assess compute cost.
"""

import os
import sys
import time
import argparse
import numpy as np

from loader   import (load_raw, build_features, all_windows,
                      expanding_windows, shrinking_windows,
                      chronological_split, dataset_summary)
from vlstm    import (train_vlstm, predict, build_sequences,
                      scale_features, find_best_lookback,
                      top_vsn_features, set_seed)
from backtest import (execute_strategy, generate_live_signal,
                      summarise_window_result, stream_consensus)
from conviction import compute_conviction
from writer   import write_outputs


# ── Config ────────────────────────────────────────────────────────────────────

TARGET_ETFS  = ["TLT", "VNQ", "SLV", "GLD", "HYG", "LQD"]
OPTIONS      = ["A", "B"]
LOSS_MODES   = ["ce", "sharpe"]
LOOKBACKS    = [30, 45, 60]
TRAIN_PCT    = 0.70
VAL_PCT      = 0.15
EPOCHS       = 80
BATCH_SIZE   = 64
HIDDEN_DIM   = 128
LSTM_LAYERS  = 2
DROPOUT      = 0.3
LR           = 5e-4
PATIENCE     = 15
FEE_BPS      = 10.0


# ── Single window trainer ─────────────────────────────────────────────────────

def train_one_window(
    window:    dict,
    df_raw,
    X_full_A:  np.ndarray,     # full dataset features option A (for live signal)
    X_full_B:  np.ndarray,     # full dataset features option B
    feat_names_A: list,
    feat_names_B: list,
    epochs:    int = EPOCHS,
    verbose:   bool = True,
) -> dict:
    """
    Train all 4 models (A_ce, A_sharpe, B_ce, B_sharpe) for one window.
    Returns the best result dict by val_sharpe.
    """
    label      = window["label"]
    start_yr   = window["start_year"]
    end_yr     = window["end_year"]
    all_results = []

    for option in OPTIONS:
        # Build features for this window slice
        try:
            feat = build_features(df_raw, option=option,
                                  start_year=start_yr, end_year=end_yr)
        except ValueError as e:
            if verbose:
                print(f"  ⚠️  {label} option={option}: {e}")
            continue

        X         = feat["X"]
        y_labels  = feat["y_labels"]
        y_returns = feat["y_returns"]
        feat_names= feat["feature_names"]
        dates     = feat["dates"]
        n_etfs    = feat["n_etfs"]
        n_features= X.shape[1]

        # Auto-select lookback
        try:
            lookback = find_best_lookback(
                X, y_labels, y_returns,
                TRAIN_PCT, VAL_PCT, n_etfs, n_features,
                candidates=LOOKBACKS, epochs=15,
            )
        except Exception:
            lookback = 45

        # Build sequences
        X_seq, y_lab, y_ret = build_sequences(X, y_labels, y_returns, lookback)
        if len(X_seq) < 100:
            continue

        # Dates aligned to sequences
        dates_seq = dates[lookback:]
        n         = len(X_seq)
        t_end     = int(n * TRAIN_PCT)
        v_end     = int(n * (TRAIN_PCT + VAL_PCT))

        X_train   = X_seq[:t_end]
        X_val     = X_seq[t_end:v_end]
        X_test    = X_seq[v_end:]
        y_train_l = y_lab[:t_end]
        y_val_l   = y_lab[t_end:v_end]
        y_test_l  = y_lab[v_end:]
        y_train_r = y_ret[:t_end]
        y_val_r   = y_ret[t_end:v_end]
        y_test_r  = y_ret[v_end:]
        dates_test= dates_seq[v_end:]

        if len(X_train) < 50 or len(X_val) < 10 or len(X_test) < 10:
            continue

        X_train_s, X_val_s, X_test_s, scale_mean, scale_std = \
            scale_features(X_train, X_val, X_test)

        for loss_mode in LOSS_MODES:
            run_label = f"{label} opt={option} loss={loss_mode}"
            if verbose:
                print(f"  🏋️  Training {run_label} ...")
            t0 = time.time()

            try:
                train_res = train_vlstm(
                    X_train_s, y_train_l, y_train_r,
                    X_val_s,   y_val_l,   y_val_r,
                    n_etfs     = n_etfs,
                    loss_mode  = loss_mode,
                    hidden_dim = HIDDEN_DIM,
                    lstm_layers= LSTM_LAYERS,
                    dropout    = DROPOUT,
                    lr         = LR,
                    epochs     = epochs,
                    batch_size = BATCH_SIZE,
                    patience   = PATIENCE,
                )
            except Exception as e:
                if verbose:
                    print(f"    ❌ Failed: {e}")
                continue

            model = train_res["model"]

            # Backtest on test set
            preds, proba, attn = predict(model, X_test_s)
            bt = execute_strategy(
                preds, proba, y_test_r, dates_test,
                TARGET_ETFS, FEE_BPS,
            )

            # Live signal — use full dataset features for this option
            X_full = X_full_A if option == "A" else X_full_B
            fn     = feat_names_A if option == "A" else feat_names_B

            live = generate_live_signal(
                model, X_full, scale_mean, scale_std,
                lookback, TARGET_ETFS, fn,
            )

            result = summarise_window_result(
                window, train_res, bt, live, lookback, option
            )
            result["option"] = option

            elapsed = time.time() - t0
            if verbose:
                print(f"    ✅ val_sharpe={train_res['val_sharpe']:.3f} "
                      f"bt_ann={bt['ann_return']*100:.1f}% "
                      f"live={live['signal']} "
                      f"({elapsed:.0f}s)")

            all_results.append(result)

    if not all_results:
        return None

    # Pick best by val_sharpe
    best = max(all_results, key=lambda r: r.get("val_sharpe", -999))
    best["all_models"] = all_results    # keep all 4 for UI
    return best


# ── Full training run ─────────────────────────────────────────────────────────

def run_full_training(df_raw, hf_token: str, epochs: int = EPOCHS):
    import pandas as pd

    print("=" * 60)
    print("P2-ETF-VLSTM-SIGNAL  |  Full Training Run")
    print("=" * 60)

    data_through = str(df_raw.index[-1].date())
    print(f"Data through: {data_through}")
    print(f"Dataset: {len(df_raw)} rows  "
          f"{df_raw.index[0].date()} → {df_raw.index[-1].date()}")

    # Pre-build full-dataset features for live signal generation
    print("\n📐 Pre-building full-dataset features for live signal...")
    try:
        full_A = build_features(df_raw, option="A", start_year=2008)
        X_full_A   = full_A["X"]
        fn_A       = full_A["feature_names"]
    except Exception as e:
        print(f"  ⚠️  Option A full features failed: {e}")
        X_full_A, fn_A = None, []

    try:
        full_B = build_features(df_raw, option="B", start_year=2008)
        X_full_B   = full_B["X"]
        fn_B       = full_B["feature_names"]
    except Exception as e:
        print(f"  ⚠️  Option B full features failed: {e}")
        X_full_B, fn_B = None, []

    # Generate all windows
    data_end       = df_raw.index.year.max()
    exp_windows    = expanding_windows(
        first_train_end=2010, data_end_year=data_end, df_raw=df_raw
    )
    shr_windows    = shrinking_windows(
        last_train_start=data_end - 1, data_end_year=data_end, df_raw=df_raw
    )

    total_windows  = len(exp_windows) + len(shr_windows)
    print(f"\n🗂️  Windows: {len(exp_windows)} expanding + "
          f"{len(shr_windows)} shrinking = {total_windows} total")
    print(f"🔧  Models per window: {len(OPTIONS) * len(LOSS_MODES)} "
          f"(A_ce, A_sharpe, B_ce, B_sharpe)")
    print(f"📊  Total training runs: {total_windows * len(OPTIONS) * len(LOSS_MODES)}")

    t_global = time.time()

    # ── Train expanding windows ───────────────────────────────────────────────
    print(f"\n{'='*40}")
    print("📈 EXPANDING STREAM")
    print(f"{'='*40}")
    exp_results = []
    for i, window in enumerate(exp_windows):
        print(f"\n[{i+1}/{len(exp_windows)}] Window: {window['label']}")
        result = train_one_window(
            window, df_raw, X_full_A, X_full_B, fn_A, fn_B, epochs=epochs
        )
        if result:
            exp_results.append(result)
        else:
            print(f"  ⚠️  No valid result for {window['label']}")

    # ── Train shrinking windows ───────────────────────────────────────────────
    print(f"\n{'='*40}")
    print("📉 SHRINKING STREAM")
    print(f"{'='*40}")
    shr_results = []
    for i, window in enumerate(shr_windows):
        print(f"\n[{i+1}/{len(shr_windows)}] Window: {window['label']}")
        result = train_one_window(
            window, df_raw, X_full_A, X_full_B, fn_A, fn_B, epochs=epochs
        )
        if result:
            shr_results.append(result)
        else:
            print(f"  ⚠️  No valid result for {window['label']}")

    total_elapsed = time.time() - t_global
    print(f"\n⏱️  Total training time: {total_elapsed/60:.1f} minutes")

    # ── Compute consensus ─────────────────────────────────────────────────────
    exp_consensus = stream_consensus(exp_results, TARGET_ETFS)
    shr_consensus = stream_consensus(shr_results, TARGET_ETFS)

    print(f"\n📊 Expanding consensus:  {exp_consensus.get('signal')} "
          f"({exp_consensus.get('strength')})")
    print(f"📊 Shrinking consensus:  {shr_consensus.get('signal')} "
          f"({shr_consensus.get('strength')})")

    # ── Write to HF ───────────────────────────────────────────────────────────
    print("\n📤 Writing outputs to HuggingFace...")
    write_outputs(
        expanding_results    = exp_results,
        shrinking_results    = shr_results,
        expanding_consensus  = exp_consensus,
        shrinking_consensus  = shr_consensus,
        data_through         = data_through,
        hf_token             = hf_token,
    )

    print("\n✅ Done.")
    return total_elapsed


# ── Benchmark mode ────────────────────────────────────────────────────────────

def run_benchmark(df_raw, epochs: int = EPOCHS):
    """
    Train a single window (2008→2020) with all 4 models.
    Print timing and extrapolate to full run.
    """
    import pandas as pd

    print("=" * 60)
    print("⏱️  BENCHMARK MODE — timing single window (2008→2020)")
    print("=" * 60)

    window = {
        "start_year": 2008,
        "end_year":   2020,
        "label":      "2008→2020",
        "stream":     "expanding",
    }

    data_end   = df_raw.index.year.max()

    try:
        full_A = build_features(df_raw, option="A", start_year=2008)
        X_full_A, fn_A = full_A["X"], full_A["feature_names"]
    except Exception as e:
        print(f"Feature build failed: {e}")
        return

    try:
        full_B = build_features(df_raw, option="B", start_year=2008)
        X_full_B, fn_B = full_B["X"], full_B["feature_names"]
    except Exception as e:
        X_full_B, fn_B = X_full_A, fn_A

    timings = {}
    for option in OPTIONS:
        try:
            feat = build_features(df_raw, option=option,
                                  start_year=2008, end_year=2020)
        except Exception as e:
            print(f"  ⚠️  option={option}: {e}")
            continue

        X, y_labels, y_returns = feat["X"], feat["y_labels"], feat["y_returns"]
        n_features = X.shape[1]
        n_etfs     = feat["n_etfs"]
        dates      = feat["dates"]

        lookback = find_best_lookback(
            X, y_labels, y_returns,
            TRAIN_PCT, VAL_PCT, n_etfs, n_features,
            candidates=LOOKBACKS, epochs=10,
        )
        print(f"\n  Option {option}: lookback={lookback}d, "
              f"n_features={n_features}")

        X_seq, y_lab, y_ret = build_sequences(X, y_labels, y_returns, lookback)
        n      = len(X_seq)
        t_end  = int(n * TRAIN_PCT)
        v_end  = int(n * (TRAIN_PCT + VAL_PCT))

        X_train_s, X_val_s, X_test_s, sm, ss = scale_features(
            X_seq[:t_end], X_seq[t_end:v_end], X_seq[v_end:]
        )
        y_train_l = y_lab[:t_end];  y_val_l = y_lab[t_end:v_end]
        y_train_r = y_ret[:t_end];  y_val_r = y_ret[t_end:v_end]

        for loss_mode in LOSS_MODES:
            print(f"  🏋️  option={option} loss={loss_mode} ... ", end="", flush=True)
            t0 = time.time()
            try:
                res = train_vlstm(
                    X_train_s, y_train_l, y_train_r,
                    X_val_s,   y_val_l,   y_val_r,
                    n_etfs=n_etfs, loss_mode=loss_mode,
                    hidden_dim=HIDDEN_DIM, lstm_layers=LSTM_LAYERS,
                    dropout=DROPOUT, lr=LR, epochs=epochs,
                    batch_size=BATCH_SIZE, patience=PATIENCE,
                )
                elapsed = time.time() - t0
                timings[f"{option}_{loss_mode}"] = elapsed
                print(f"{elapsed:.0f}s  val_sharpe={res['val_sharpe']:.3f} "
                      f"epochs={res['epochs_run']}")
            except Exception as e:
                elapsed = time.time() - t0
                print(f"FAILED ({e})")

    # ── Extrapolation ─────────────────────────────────────────────────────────
    if timings:
        per_window = sum(timings.values())
        data_end   = df_raw.index.year.max()
        n_exp      = data_end - 2010 + 1     # expanding windows
        n_shr      = data_end - 2008         # shrinking windows
        total_windows = n_exp + n_shr
        total_est     = per_window * total_windows

        print("\n" + "=" * 60)
        print("📊 BENCHMARK RESULTS")
        print("=" * 60)
        for k, v in timings.items():
            print(f"  {k:20s}: {v:.0f}s ({v/60:.1f} min)")
        print(f"\n  Per window (4 models): {per_window:.0f}s ({per_window/60:.1f} min)")
        print(f"  Total windows:         {total_windows} "
              f"({n_exp} expanding + {n_shr} shrinking)")
        print(f"  Estimated full run:    {total_est:.0f}s "
              f"({total_est/60:.1f} min / {total_est/3600:.2f} hrs)")
        print(f"\n  GitHub Actions free tier: 2000 min/month")
        print(f"  This run × 22 trading days: "
              f"{total_est/60*22:.0f} min/month")

        if total_est/60 * 22 <= 2000:
            print("  ✅ WITHIN free tier limit")
        else:
            overage = total_est/60 * 22 - 2000
            print(f"  ⚠️  EXCEEDS free tier by {overage:.0f} min/month")
            print("  💡 Consider: weekly full retrain + daily latest-window only")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="P2-ETF-VLSTM-SIGNAL trainer")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark timing only (no HF write)")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Max training epochs (default {EPOCHS})")
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        print("❌ HF_TOKEN environment variable not set.")
        sys.exit(1)

    print("📡 Loading dataset from HuggingFace...")
    df_raw = load_raw(hf_token)
    summary = dataset_summary(df_raw)
    print(f"   Rows: {summary['rows']:,}  |  "
          f"{summary['start_date']} → {summary['end_date']}")
    print(f"   ETFs: {summary['etfs']}")
    print(f"   Macro: {summary['macro']}")

    if args.benchmark:
        run_benchmark(df_raw, epochs=args.epochs)
    else:
        run_full_training(df_raw, hf_token, epochs=args.epochs)


if __name__ == "__main__":
    # Fix for the placeholder used above — import pandas properly
    import pandas as pd
    main()
