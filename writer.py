"""
writer.py
P2-ETF-VLSTM-SIGNAL

Writes training results to HuggingFace output dataset.
Supports prefixed filenames for multiple universes (fi, equity) in same dataset.
"""

import io
import json
import traceback
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

# Default dataset (FI universe) – kept for backward compatibility
DEFAULT_DATASET = "P2SAMAPA/p2-etf-vlstm-outputs"

# Keys that contain non-serialisable objects or circular refs — always strip
_STRIP_KEYS = {"equity", "daily_rets", "etf_held", "dates", "all_models"}

# ── JSON serialiser ───────────────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj) if not np.isnan(obj) else None
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, pd.Timestamp): return str(obj.date())
        return super().default(obj)

def _serialise(obj) -> str:
    return json.dumps(obj, cls=_NumpyEncoder, indent=2)

# ── Result sanitiser — iterative, no recursion ────────────────────────────────

def _sanitise(r: dict) -> dict:
    """
    Flatten a result dict to JSON-safe primitives.
    Uses an explicit stack instead of recursion to avoid hitting
    Python's recursion limit on deeply nested structures.
    """
    def _clean_value(v):
        if isinstance(v, pd.Series):
            return None
        if isinstance(v, np.ndarray):
            arr = v.tolist()
            return _clean_list(arr)
        if isinstance(v, float):
            if np.isnan(v) or np.isinf(v):
                return None
            return v
        if isinstance(v, np.floating):
            f = float(v)
            return None if (np.isnan(f) or np.isinf(f)) else f
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, dict):
            return _clean_dict(v)
        if isinstance(v, list):
            return _clean_list(v)
        return v

    def _clean_list(lst):
        out = []
        for item in lst:
            out.append(_clean_value(item))
        return out

    def _clean_dict(d):
        out = {}
        for k, v in d.items():
            if k in _STRIP_KEYS:
                continue
            out[k] = _clean_value(v)
        return out

    return _clean_dict(r)

# ── Stream writer ─────────────────────────────────────────────────────────────

def write_stream(
    stream: str,
    results: list,
    consensus: dict,
    data_through: str,
    hf_token: str,
    dataset_name: str = DEFAULT_DATASET,
    file_prefix: str = "fi",  # NEW: prefix for filenames (fi_ or equity_)
) -> bool:
    """
    Write one stream's results to its own file on HuggingFace.
    expanding → {prefix}_expanding_latest.json
    shrinking → {prefix}_shrinking_latest.json
    """
    assert stream in ("expanding", "shrinking")

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        est_now = datetime.now(timezone.utc) - timedelta(hours=5)
        run_date = est_now.strftime("%Y-%m-%d")

        # Sanitise all window results
        clean_windows = []
        for r in results:
            if r is None:
                continue
            try:
                clean_windows.append(_sanitise(r))
            except Exception as e:
                print(f" ⚠️ Could not sanitise window result: {e}")

        payload = {
            "stream": stream,
            "run_date": run_date,
            "data_through": data_through,
            "consensus": consensus,
            "windows": clean_windows,
        }

        # PREFIXED filename: fi_expanding_latest.json or equity_expanding_latest.json
        filename = f"{file_prefix}_{stream}_latest.json"

        # Verify serialisable before uploading
        json_bytes = _serialise(payload).encode()

        api.upload_file(
            path_or_fileobj=io.BytesIO(json_bytes),
            path_in_repo=filename,
            repo_id=dataset_name,
            repo_type="dataset",
            commit_message=f"[{stream}] Update signals — {run_date}",
        )

        _append_history(api, run_date, data_through, stream, consensus, dataset_name, file_prefix)

        print(f"✅ {filename} written to {dataset_name} for {run_date}")
        return True

    except Exception as e:
        print(f"❌ Failed to write {stream} outputs: {e}")
        traceback.print_exc()
        return False

# ── History appender ──────────────────────────────────────────────────────────

def _append_history(api, run_date, data_through, stream, consensus, dataset_name: str, file_prefix: str = "fi"):
    prefix = "exp" if stream == "expanding" else "shr"
    new_row = {
        "run_date": run_date,
        "data_through": data_through,
        f"{prefix}_signal": consensus.get("signal", ""),
        f"{prefix}_strength": consensus.get("strength", ""),
        f"{prefix}_score_pct": consensus.get("score_pct", 0),
    }

    # PREFIXED history filename: fi_history.parquet or equity_history.parquet
    history_file = f"{file_prefix}_history.parquet"
    
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id=dataset_name, filename=history_file,
            repo_type="dataset", token=api.token, force_download=True,
        )
        existing = pd.read_parquet(local)
        if run_date in existing["run_date"].values:
            for col, val in new_row.items():
                existing.loc[existing["run_date"] == run_date, col] = val
            combined = existing
        else:
            combined = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
    except Exception:
        combined = pd.DataFrame([new_row])

    buf = io.BytesIO()
    combined.to_parquet(buf, index=False)
    buf.seek(0)
    api.upload_file(
        path_or_fileobj=buf,
        path_in_repo=history_file,
        repo_id=dataset_name,
        repo_type="dataset",
        commit_message=f"[{stream}] Append history — {run_date}",
    )

# ── Readers (used by Streamlit) ───────────────────────────────────────────────

def load_stream(stream: str, hf_token: str, dataset_name: str = DEFAULT_DATASET, file_prefix: str = "fi") -> dict:
    """Load the latest JSON file for a given stream from the specified dataset."""
    assert stream in ("expanding", "shrinking")
    filename = f"{file_prefix}_{stream}_latest.json"  # prefixed
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id=dataset_name, filename=filename,
            repo_type="dataset", token=hf_token, force_download=True,
        )
        with open(local) as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Could not load {filename} from {dataset_name}: {e}")
        return {}

def load_history(hf_token: str, dataset_name: str = DEFAULT_DATASET, file_prefix: str = "fi") -> pd.DataFrame:
    """Load the history.parquet file from the specified dataset."""
    filename = f"{file_prefix}_history.parquet"  # prefixed
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id=dataset_name, filename=filename,
            repo_type="dataset", token=hf_token, force_download=True,
        )
        return pd.read_parquet(local)
    except Exception as e:
        print(f"⚠️ Could not load {filename} from {dataset_name}: {e}")
        return pd.DataFrame()

def load_latest(hf_token: str, dataset_name: str = DEFAULT_DATASET, file_prefix: str = "fi") -> dict:
    """
    Assembles both stream files into a single dict for app.py.
    Uses file_prefix to load correct universe files.
    """
    exp = load_stream("expanding", hf_token, dataset_name, file_prefix)
    shr = load_stream("shrinking", hf_token, dataset_name, file_prefix)

    has_exp = bool(exp)
    has_shr = bool(shr)

    if not has_exp and not has_shr:
        return {}

    source = exp if has_exp else shr

    return {
        "run_date": source.get("run_date", "—"),
        "data_through": source.get("data_through", "—"),
        "expanding": {"consensus": exp.get("consensus", {}) if has_exp else {},
                      "windows": exp.get("windows", []) if has_exp else []},
        "shrinking": {"consensus": shr.get("consensus", {}) if has_shr else {},
                      "windows": shr.get("windows", []) if has_shr else []},
    }
