"""
writer.py
"""

import io
import json
import traceback
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

DEFAULT_DATASET = "P2SAMAPA/p2-etf-vlstm-outputs"
_STRIP_KEYS = {"model", "daily_rets", "etf_held", "dates", "all_models"}

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj) if not np.isnan(obj) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj.date())
        try:
            import torch
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().tolist()
        except ImportError:
            pass
        return super().default(obj)

def _serialise(obj) -> str:
    return json.dumps(obj, cls=_NumpyEncoder, indent=2)

def _sanitise(r: dict, target_etfs: list = None) -> dict:
    def _clean_value(v):
        if isinstance(v, pd.Series):
            return None
        if isinstance(v, pd.DataFrame):
            return None
        if isinstance(v, np.ndarray):
            return _clean_list(v.tolist())
        if isinstance(v, float):
            return None if (np.isnan(v) or np.isinf(v)) else v
        if isinstance(v, np.floating):
            f = float(v)
            return None if (np.isnan(f) or np.isinf(f)) else f
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, dict):
            return _clean_dict(v)
        if isinstance(v, list):
            return _clean_list(v)
        if isinstance(v, tuple):
            return _clean_list(list(v))
        try:
            import torch
            if isinstance(obj, torch.Tensor):
                return v.detach().cpu().tolist()
        except (ImportError, NameError):
            pass
        return v

    def _clean_list(lst):
        return [_clean_value(item) for item in lst]

    def _clean_dict(d):
        return {k: _clean_value(v) for k, v in d.items() if k not in _STRIP_KEYS}

    return _clean_dict(r)

def write_stream(
    stream: str,
    results: list,
    consensus: dict,
    data_through: str,
    hf_token: str,
    dataset_name: str = DEFAULT_DATASET,
    file_prefix: str = "fi",
    target_etfs: list = None,
) -> bool:
    assert stream in ("expanding", "shrinking")

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)

        est_now = datetime.now(timezone.utc) - timedelta(hours=5)
        run_date = est_now.strftime("%Y-%m-%d")

        clean_windows = []
        for r in results:
            if r is None:
                continue
            try:
                clean_windows.append(_sanitise(r, target_etfs=target_etfs))
            except Exception as e:
                print(f" ⚠️ Could not sanitise window result: {e}")
                traceback.print_exc()

        # CRITICAL FIX: Create a copy of consensus to avoid modifying original
        clean_consensus = _normalise_consensus(consensus.copy(), target_etfs)

        payload = {
            "stream": stream,
            "run_date": run_date,
            "data_through": data_through,
            "consensus": clean_consensus,
            "windows": clean_windows,
            "universe": file_prefix,
            "target_etfs": target_etfs,
        }

        filename = f"{file_prefix}_{stream}_latest.json"

        try:
            json_bytes = _serialise(payload).encode()
            print(f" 📦 Payload: {len(json_bytes)/1024:.1f} KB | "
                  f"{len(clean_windows)} windows | universe: {file_prefix}")
        except Exception as e:
            print(f"❌ Serialisation failed for {filename}: {e}")
            traceback.print_exc()
            return False

        api.upload_file(
            path_or_fileobj=io.BytesIO(json_bytes),
            path_in_repo=filename,
            repo_id=dataset_name,
            repo_type="dataset",
            commit_message=f"[{stream}] Update signals — {run_date} ({file_prefix})",
        )

        _append_history(api, run_date, data_through, stream,
                       clean_consensus, dataset_name, file_prefix, target_etfs)

        print(f"✅ {filename} written to {dataset_name} for {run_date}")
        return True

    except Exception as e:
        print(f"❌ Failed to write {stream} outputs: {e}")
        traceback.print_exc()
        return False

def _normalise_consensus(consensus: dict, target_etfs: list = None) -> dict:
    """
    CRITICAL FIX: Don't modify original dict, create new one.
    """
    if not isinstance(consensus, dict):
        return consensus

    # Create NEW dict, don't modify original
    normalised = {}
    
    # Copy non-ETF keys first
    for key, value in consensus.items():
        if target_etfs and key in target_etfs:
            continue  # Skip ETF vote keys for now
        normalised[key] = value

    # Add votes dict if we have target_etfs
    if target_etfs:
        votes = {}
        for etf in target_etfs:
            if etf in consensus:
                votes[etf] = consensus[etf]
        if votes:
            normalised["votes"] = votes

    # Ensure required keys exist
    if "signal" not in normalised:
        normalised["signal"] = ""
    if "strength" not in normalised:
        normalised["strength"] = ""
    if "score_pct" not in normalised:
        normalised["score_pct"] = 0
        
    return normalised

def _append_history(api, run_date, data_through, stream, consensus,
                   dataset_name: str, file_prefix: str = "fi",
                   target_etfs: list = None):
    prefix = "exp" if stream == "expanding" else "shr"
    
    new_row = {
        "run_date": run_date,
        "data_through": data_through,
        f"{prefix}_signal": consensus.get("signal", ""),
        f"{prefix}_strength": consensus.get("strength", ""),
        f"{prefix}_score_pct": consensus.get("score_pct", 0),
        f"{prefix}_agreement": consensus.get("agreement", 0),
        "universe": file_prefix,
    }

    history_file = f"{file_prefix}_history.parquet"

    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id=dataset_name, filename=history_file,
            repo_type="dataset", token=api.token, force_download=True,
        )
        existing = pd.read_parquet(local)
        
        for col in new_row.keys():
            if col not in existing.columns:
                existing[col] = None
                
        if run_date in existing["run_date"].values:
            for col, val in new_row.items():
                existing.loc[existing["run_date"] == run_date, col] = val
            combined = existing
        else:
            combined = pd.concat([existing, pd.DataFrame([new_row])],
                               ignore_index=True)
    except Exception as e:
        print(f" ⚠️ Creating new history file {history_file}: {e}")
        combined = pd.DataFrame([new_row])

    buf = io.BytesIO()
    combined.to_parquet(buf, index=False)
    buf.seek(0)
    api.upload_file(
        path_or_fileobj=buf,
        path_in_repo=history_file,
        repo_id=dataset_name,
        repo_type="dataset",
        commit_message=f"[{stream}] Append history — {run_date} ({file_prefix})",
    )
    print(f" 📚 Updated {history_file}: {len(combined)} rows")

def load_stream(stream: str, hf_token: str,
               dataset_name: str = DEFAULT_DATASET,
               file_prefix: str = "fi") -> dict:
    assert stream in ("expanding", "shrinking")
    filename = f"{file_prefix}_{stream}_latest.json"
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

def load_history(hf_token: str,
                dataset_name: str = DEFAULT_DATASET,
                file_prefix: str = "fi") -> pd.DataFrame:
    filename = f"{file_prefix}_history.parquet"
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

def load_latest(hf_token: str,
               dataset_name: str = DEFAULT_DATASET,
               file_prefix: str = "fi") -> dict:
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
        "universe": file_prefix,
        "target_etfs": source.get("target_etfs", []),
        "expanding": {
            "consensus": exp.get("consensus", {}) if has_exp else {},
            "windows": exp.get("windows", []) if has_exp else [],
        },
        "shrinking": {
            "consensus": shr.get("consensus", {}) if has_shr else {},
            "windows": shr.get("windows", []) if has_shr else [],
        },
    }
