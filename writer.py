"""
writer.py
P2-ETF-VLSTM-SIGNAL

Writes training results to the HuggingFace output dataset:
  P2SAMAPA/p2-etf-vlstm-outputs

Two separate files — one per stream — so expanding and shrinking jobs
can run in parallel without any merge/race condition:
  expanding_latest.json   → written by retrain_expanding.yml
  shrinking_latest.json   → written by retrain_shrinking.yml

History:
  history.parquet         → one row per run date, appended by whichever
                            stream finishes last (last-write-wins is fine
                            since both write the same consensus columns)
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


# ── Stream writer — one file per stream ───────────────────────────────────────

def write_stream(
    stream:     str,            # "expanding" or "shrinking"
    results:    list,
    consensus:  dict,
    data_through: str,
    hf_token:   str,
) -> bool:
    """
    Write one stream's results to its own file on HuggingFace.
    expanding → expanding_latest.json
    shrinking → shrinking_latest.json

    Each file is self-contained — Streamlit reads them independently.
    No coordination between the two jobs required.
    """
    assert stream in ("expanding", "shrinking"), "stream must be 'expanding' or 'shrinking'"

    try:
        from huggingface_hub import HfApi
        api      = HfApi(token=hf_token)
        est_now  = datetime.now(timezone.utc) - timedelta(hours=5)
        run_date = est_now.strftime("%Y-%m-%d")

        payload = {
            "stream":       stream,
            "run_date":     run_date,
            "data_through": data_through,
            "consensus":    consensus,
            "windows":      [_sanitise(r) for r in results if r],
        }

        filename = f"{stream}_latest.json"
        api.upload_file(
            path_or_fileobj=io.BytesIO(_serialise(payload).encode()),
            path_in_repo=filename,
            repo_id=OUTPUT_DATASET,
            repo_type="dataset",
            commit_message=f"[{stream}] Update signals — {run_date}",
        )

        _append_history(api, run_date, data_through, stream, consensus)

        print(f"✅ {filename} written to {OUTPUT_DATASET} for {run_date}")
        return True

    except Exception as e:
        print(f"❌ Failed to write {stream} outputs: {e}")
        traceback.print_exc()
        return False


# ── History appender ──────────────────────────────────────────────────────────

def _append_history(
    api,
    run_date:    str,
    data_through: str,
    stream:      str,
    consensus:   dict,
):
    """
    Append/update one row in history.parquet for this run_date.
    Each stream writes its own columns — last-write-wins on the shared row
    is safe because expanding and shrinking write different columns.
    """
    prefix  = "exp" if stream == "expanding" else "shr"
    new_row = {
        "run_date":              run_date,
        "data_through":          data_through,
        f"{prefix}_signal":      consensus.get("signal", ""),
        f"{prefix}_strength":    consensus.get("strength", ""),
        f"{prefix}_score_pct":   consensus.get("score_pct", 0),
    }

    try:
        from huggingface_hub import hf_hub_download
        local    = hf_hub_download(
            repo_id=OUTPUT_DATASET, filename="history.parquet",
            repo_type="dataset", token=api.token,
        )
        existing = pd.read_parquet(local)

        # Upsert: update existing row for this run_date, or append new one
        if run_date in existing["run_date"].values:
            for col, val in new_row.items():
                existing.loc[existing["run_date"] == run_date, col] = val
            combined = existing
        else:
            combined = pd.concat(
                [existing, pd.DataFrame([new_row])], ignore_index=True
            )
    except Exception:
        combined = pd.DataFrame([new_row])

    buf = io.BytesIO()
    combined.to_parquet(buf, index=False)
    buf.seek(0)

    api.upload_file(
        path_or_fileobj=buf,
        path_in_repo="history.parquet",
        repo_id=OUTPUT_DATASET,
        repo_type="dataset",
        commit_message=f"[{stream}] Append history — {run_date}",
    )


# ── Readers (used by Streamlit) ───────────────────────────────────────────────

def load_stream(stream: str, hf_token: str) -> dict:
    """
    Load expanding_latest.json or shrinking_latest.json from HF.
    Returns empty dict if not yet available.
    """
    assert stream in ("expanding", "shrinking")
    filename = f"{stream}_latest.json"
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id=OUTPUT_DATASET, filename=filename,
            repo_type="dataset", token=hf_token, force_download=True,
        )
        with open(local) as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Could not load {filename}: {e}")
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
        print(f"⚠️  Could not load history.parquet: {e}")
        return pd.DataFrame()


# ── Legacy single-file loader (backwards compat) ─────────────────────────────

def load_latest(hf_token: str) -> dict:
    """
    Backwards-compatible loader — assembles the old single-dict format
    from the two separate stream files so app.py works unchanged.
    """
    exp = load_stream("expanding", hf_token)
    shr = load_stream("shrinking", hf_token)

    if not exp and not shr:
        return {}

    return {
        "run_date":    (exp or shr).get("run_date", "—"),
        "data_through":(exp or shr).get("data_through", "—"),
        "expanding":   {"consensus": exp.get("consensus", {}),
                        "windows":   exp.get("windows", [])},
        "shrinking":   {"consensus": shr.get("consensus", {}),
                        "windows":   shr.get("windows", [])},
    }
