"""
writer.py
P2-ETF-VLSTM-SIGNAL

Writes training results and live signals to the HuggingFace output dataset:
  P2SAMAPA/p2-etf-vlstm-outputs

Two files maintained:
  latest.json     → today's full results (overwritten daily)
  history.parquet → one row per run date appended to history
"""

import io
import json
import traceback
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd


OUTPUT_DATASET = "P2SAMAPA/p2-etf-vlstm-outputs"


# ── JSON serialiser ───────────────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):   return int(obj)
        if isinstance(obj, np.floating):  return float(obj)
        if isinstance(obj, np.ndarray):   return obj.tolist()
        if isinstance(obj, pd.Timestamp): return str(obj.date())
        return super().default(obj)


def _serialise(obj) -> str:
    return json.dumps(obj, cls=_NumpyEncoder, indent=2)


# ── Result sanitiser ──────────────────────────────────────────────────────────

def _sanitise(r: dict) -> dict:
    """Strip non-serialisable objects before JSON writing."""
    skip = {"equity", "daily_rets", "etf_held", "dates"}
    out  = {}
    for k, v in r.items():
        if k in skip:
            continue
        if isinstance(v, pd.Series):
            continue
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = _sanitise(v)
        elif isinstance(v, list):
            out[k] = [_sanitise(i) if isinstance(i, dict) else i for i in v]
        else:
            out[k] = v
    return out


# ── Main write function ───────────────────────────────────────────────────────

def write_outputs(
    expanding_results:   list,
    shrinking_results:   list,
    expanding_consensus: dict,
    shrinking_consensus: dict,
    data_through:        str,
    hf_token:            str,
    benchmark_mode:      bool = False,
    benchmark_timing:    dict = None,
) -> bool:
    """Write full results to HuggingFace. Returns True on success."""
    try:
        from huggingface_hub import HfApi
        api      = HfApi(token=hf_token)
        est_now  = datetime.now(timezone.utc) - timedelta(hours=5)
        run_date = est_now.strftime("%Y-%m-%d")

        payload = {
            "run_date":       run_date,
            "data_through":   data_through,
            "benchmark_mode": benchmark_mode,
            "expanding": {
                "consensus": expanding_consensus,
                "windows":   [_sanitise(r) for r in expanding_results if r],
            },
            "shrinking": {
                "consensus": shrinking_consensus,
                "windows":   [_sanitise(r) for r in shrinking_results if r],
            },
        }

        if benchmark_mode and benchmark_timing:
            payload["benchmark_timing"] = benchmark_timing

        api.upload_file(
            path_or_fileobj=io.BytesIO(_serialise(payload).encode()),
            path_in_repo="latest.json",
            repo_id=OUTPUT_DATASET,
            repo_type="dataset",
            commit_message=f"Update signals — {run_date}",
        )

        _append_history(api, run_date, data_through,
                        expanding_consensus, shrinking_consensus)

        print(f"✅ Outputs written to {OUTPUT_DATASET} for {run_date}")
        return True

    except Exception as e:
        print(f"❌ Failed to write outputs: {e}")
        traceback.print_exc()
        return False


# ── History appender ──────────────────────────────────────────────────────────

def _append_history(api, run_date, data_through,
                    expanding_consensus, shrinking_consensus):
    new_row = pd.DataFrame([{
        "run_date":      run_date,
        "data_through":  data_through,
        "exp_signal":    expanding_consensus.get("signal", ""),
        "exp_strength":  expanding_consensus.get("strength", ""),
        "exp_score_pct": expanding_consensus.get("score_pct", 0),
        "shr_signal":    shrinking_consensus.get("signal", ""),
        "shr_strength":  shrinking_consensus.get("strength", ""),
        "shr_score_pct": shrinking_consensus.get("score_pct", 0),
    }])

    try:
        from huggingface_hub import hf_hub_download
        local    = hf_hub_download(
            repo_id=OUTPUT_DATASET, filename="history.parquet",
            repo_type="dataset", token=api.token,
        )
        existing = pd.read_parquet(local)
        existing = existing[existing["run_date"] != run_date]
        combined = pd.concat([existing, new_row], ignore_index=True)
    except Exception:
        combined = new_row

    buf = io.BytesIO()
    combined.to_parquet(buf, index=False)
    buf.seek(0)

    api.upload_file(
        path_or_fileobj=buf,
        path_in_repo="history.parquet",
        repo_id=OUTPUT_DATASET,
        repo_type="dataset",
        commit_message=f"Append history — {run_date}",
    )


# ── Readers (used by Streamlit) ───────────────────────────────────────────────

def load_latest(hf_token: str) -> dict:
    """Load latest.json from HF output dataset."""
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id=OUTPUT_DATASET, filename="latest.json",
            repo_type="dataset", token=hf_token, force_download=True,
        )
        with open(local) as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Could not load latest.json: {e}")
        return {}


def load_history(hf_token: str) -> pd.DataFrame:
    """Load history.parquet from HF output dataset."""
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id=OUTPUT_DATASET, filename="history.parquet",
            repo_type="dataset", token=hf_token, force_download=True,
        )
        return pd.read_parquet(local)
    except Exception as e:
        print(f"⚠️ Could not load history.parquet: {e}")
        return pd.DataFrame()
