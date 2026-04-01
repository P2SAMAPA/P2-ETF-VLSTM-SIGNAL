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
       e. Generate live next-day signal (today)
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
import pandas as pd

from loader   import (load_raw, build_features, all_windows,
                      expanding_windows, shrinking_windows,
                      chronological_split, dataset_summary,
                      DEFAULT_MACRO_COLS)
from vlstm    import (train_vlstm, predict, build_sequences,
                      scale_features, find_best_lookback,
                      top_vsn_features, set_seed)
from backtest import (execute_strategy, generate_live_signal,
                      summarise_window_result, stream_consensus)
from conviction import compute_conviction
from writer   import write_stream
from config   import get_config


# ── Config ────────────────────────────────────────────────────────────────────

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
    window:          dict,
    df_raw,
    X_full_A:        np.ndarray,
    X_full_B:        np.ndarray,
    feat_names_A:    list,
    feat_names_B:    list,
    target_etfs:     list,
    n_etfs:          int,
    feature_tickers: list,
    macro_cols:      list,
    epochs:          int = EPOCHS,
    verbose:         bool = True,
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
        try:
            feat = build_features(
                df_raw, option=option,
                start_year=start_yr, end_year=end_yr,
                target_etfs=target_etfs,
                feature_tickers=feature_tickers,
                macro_cols=macro_cols
            )
        except ValueError as e:
            if verbose:
                print(f"  ⚠️  {label} option={option}: {e}")
            continue

        X         = feat["X"]
        y_labels  = feat["y_labels"]
        y_returns = feat["y_returns"]
        feat_names= feat["feature_names"]
        dates     = feat["dates"]
        n_feat    = X.shape[1]

        try:
            lookback = find_best_lookback(
                X, y_labels, y_returns,
                TRAIN_PCT, VAL_PCT, n_etfs, n_feat,
                candidates=LOOKBACKS, epochs=15,
            )
        except Exception:
            lookback = 45

        X_seq, y_lab, y_ret = build_sequences(X, y_labels, y_returns, lookback)
        if len(X_seq) < 100:
            continue

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

            preds, proba, attn = predict(model, X_test_s)
            bt = execute_strategy(
                preds, proba, y_test_r, dates_test,
                target_etfs, FEE_BPS,
            )

            X_full = X_full_A if option == "A" else X_full_B
            fn     = feat_names_A if option == "A" else feat_names_B

            live = generate_live_signal(
                model, X_full, scale_mean, scale_std,
                lookback, target_etfs, fn,
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

    best = max(all_results, key=lambda r: r.get("val_sharpe", -999))
    best["all_models"] = all_results
    return best


# ── Stream training run ───────────────────────────────────────────────────────

def run_stream(
    stream:          str,
    df_raw,
    hf_token:        str,
    target_etfs:     list,
    n_etfs:          int,
    feature_tickers: list,
    output_dataset:  str,
    file_prefix:     str = "fi",
    macro_cols:      list = None,
    epochs:          int = EPOCHS,
):
    """
    Train all windows for one stream only and write its results to HF.
    Called by retrain_expanding.yml and retrain_shrinking.yml in parallel.
    Writes {prefix}_expanding_latest.json or {prefix}_shrinking_latest.json independently.
    """
    assert stream in ("expanding", "shrinking")
    if macro_cols is None:
        macro_cols = DEFAULT_MACRO_COLS

    icon = "📈" if stream == "expanding" else "📉"

    print("=" * 60)
    print(f"P2-ETF-VLSTM-SIGNAL  |  {stream.upper()} STREAM")
    print(f"Universe: {target_etfs}")
    print(f"File prefix: {file_prefix}")
    print("=" * 60)

    data_through = str(df_raw.index[-1].date())
    data_end     = df_raw.index.year.max()
    print(f"Data through: {data_through}")
    print(f"Dataset: {len(df_raw)} rows  "
          f"{df_raw.index[0].date()} → {df_raw.index[-1].date()}")

    print("\n📐 Pre-building full-dataset features for live signal...")
    try:
        full_A = build_features(
            df_raw, option="A", start_year=2008,
            target_etfs=target_etfs,
            feature_tickers=feature_tickers,
            macro_cols=macro_cols
        )
        X_full_A = full_A["X"]
        fn_A     = full_A["feature_names"]
    except Exception as e:
        print(f"  ⚠️  Option A full features failed: {e}")
        X_full_A, fn_A = None, []

    try:
        full_B = build_features(
            df_raw, option="B", start_year=2008,
            target_etfs=target_etfs,
            feature_tickers=feature_tickers,
            macro_cols=macro_cols
        )
        X_full_B = full_B["X"]
        fn_B     = full_B["feature_names"]
    except Exception as e:
        print(f"  ⚠️  Option B full features failed: {e}")
        X_full_B, fn_B = None, []

    if stream == "expanding":
        windows = expanding_windows(
            first_train_end=2011, data_end_year=data_end, df_raw=df_raw, step=3
        )
    else:
        windows = shrinking_windows(
            data_end_year=data_end, df_raw=df_raw, step=3
        )

    print(f"\n🗂️  {icon} {len(windows)} {stream} windows")
    print(f"🔧  Models per window: {len(OPTIONS) * len(LOSS_MODES)} "
          f"(A_ce, A_sharpe, B_ce, B_sharpe)")
    print(f"📊  Total training runs: {len(windows) * len(OPTIONS) * len(LOSS_MODES)}")

    t_global = time.time()
    results  = []

    for i, window in enumerate(windows):
        print(f"\n[{i+1}/{len(windows)}] Window: {window['label']}")
        result = train_one_window(
            window, df_raw, X_full_A, X_full_B, fn_A, fn_B,
            target_etfs, n_etfs, feature_tickers, macro_cols,
            epochs=epochs
        )
        if result:
            results.append(result)
        else:
            print(f"  ⚠️  No valid result for {window['label']}")

    total_elapsed = time.time() - t_global
    print(f"\n⏱️  {stream.capitalize()} training time: {total_elapsed/60:.1f} minutes")

    consensus = stream_consensus(results, target_etfs)
    print(f"\n📊 {stream.capitalize()} consensus: {consensus.get('signal')} "
          f"({consensus.get('strength')})")

    print(f"\n📤 Writing {file_prefix}_{stream}_latest.json to {output_dataset}...")
    write_stream(
        stream       = stream,
        results      = results,
        consensus    = consensus,
        data_through = data_through,
        hf_token     = hf_token,
        dataset_name = output_dataset,
        file_prefix  = file_prefix,
    )

    print("\n✅ Done.")
    return total_elapsed


# ── Benchmark mode ────────────────────────────────────────────────────────────

def run_benchmark(df_raw, target_etfs, n_etfs, feature_tickers, macro_cols,
                  epochs: int = EPOCHS):
    """
    Train a single window (2008→2020) with all 4 models.
    Print timing and extrapolate to full run.
    """
    print("=" * 60)
    print("⏱️  BENCHMARK MODE — timing single window (2008→2020)")
    print(f"Universe: {target_etfs}")
    print("=" * 60)

    window = {
        "start_year": 2008,
        "end_year":   2020,
        "label":      "2008→2020",
        "stream":     "expanding",
    }

    try:
        full_A = build_features(
            df_raw, option="A", start_year=2008,
            target_etfs=target_etfs,
            feature_tickers=feature_tickers,
            macro_cols=macro_cols
        )
        X_full_A, fn_A = full_A["X"], full_A["feature_names"]
    except Exception as e:
        print(f"Feature build failed: {e}")
        return

    try:
        full_B = build_features(
            df_raw, option="B", start_year=2008,
            target_etfs=target_etfs,
            feature_tickers=feature_tickers,
            macro_cols=macro_cols
        )
        X_full_B, fn_B = full_B["X"], full_B["feature_names"]
    except Exception as e:
        X_full_B, fn_B = X_full_A, fn_A

    timings = {}
    for option in OPTIONS:
        try:
            feat = build_features(
                df_raw, option=option,
                start_year=2008, end_year=2020,
                target_etfs=target_etfs,
                feature_tickers=feature_tickers,
                macro_cols=macro_cols
            )
        except Exception as e:
            print(f"  ⚠️  option={option}: {e}")
            continue

        X, y_labels, y_returns = feat["X"], feat["y_labels"], feat["y_returns"]
        n_features = X.shape[1]

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

    if timings:
        per_window = sum(timings.values())
        data_end   = df_raw.index.year.max()
        n_exp      = data_end - 2010 + 1
        n_shr      = data_end - 2008
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
    parser.add_argument("--universe", type=str, default="fi",
                        choices=["fi", "equity"],
                        help="ETF universe: 'fi' (fixed income, default) or 'equity'")
    parser.add_argument("--stream", type=str, default="both",
                        choices=["expanding", "shrinking", "both"],
                        help="Which stream to train (default: both)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark timing only (no HF write)")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Max training epochs (default {EPOCHS})")
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        print("❌ HF_TOKEN environment variable not set.")
        sys.exit(1)

    config = get_config(args.universe)
    target_etfs = config["target_etfs"]
    n_etfs = config["n_etfs"]
    feature_tickers = config["feature_tickers"]
    output_dataset = config["output_dataset"]
    file_prefix = config.get("file_prefix", "fi")

    print(f"\n🌍 Using universe: {args.universe}")
    print(f"   Target ETFs: {target_etfs}")
    print(f"   Feature tickers: {feature_tickers}")
    print(f"   Output dataset: {output_dataset}")
    print(f"   File prefix: {file_prefix}\n")

    print("📡 Loading dataset from HuggingFace...")
    df_raw = load_raw(hf_token)
    summary = dataset_summary(df_raw, target_etfs=target_etfs)
    print(f"   Rows: {summary['rows']:,}  |  "
          f"{summary['start_date']} → {summary['end_date']}")
    print(f"   ETFs: {summary['etfs']}")
    print(f"   Macro: {summary['macro']}")

    if args.benchmark:
        run_benchmark(df_raw, target_etfs, n_etfs, feature_tickers,
                      DEFAULT_MACRO_COLS, epochs=args.epochs)
    elif args.stream == "both":
        run_stream("expanding", df_raw, hf_token,
                   target_etfs, n_etfs, feature_tickers,
                   output_dataset, file_prefix, DEFAULT_MACRO_COLS, epochs=args.epochs)
        run_stream("shrinking", df_raw, hf_token,
                   target_etfs, n_etfs, feature_tickers,
                   output_dataset, file_prefix, DEFAULT_MACRO_COLS, epochs=args.epochs)
    else:
        run_stream(args.stream, df_raw, hf_token,
                   target_etfs, n_etfs, feature_tickers,
                   output_dataset, file_prefix, DEFAULT_MACRO_COLS, epochs=args.epochs)


if __name__ == "__main__":
    main()
