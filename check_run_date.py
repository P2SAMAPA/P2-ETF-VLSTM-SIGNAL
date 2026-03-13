"""
check_run_date.py
P2-ETF-VLSTM-SIGNAL

Called at the start of each retrain workflow.
Reads the stream's existing output file from HF and checks if it
was already run today (EST). Exits with code 0 if already run
(workflow should skip), code 1 if not yet run (workflow should proceed).

Usage:
  python check_run_date.py --stream expanding
  python check_run_date.py --stream shrinking
"""

import sys
import argparse
import os
from datetime import datetime, timezone, timedelta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", required=True,
                        choices=["expanding", "shrinking"])
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        print("⚠️  No HF_TOKEN — cannot check run date, proceeding.")
        sys.exit(1)   # proceed

    est_now  = datetime.now(timezone.utc) - timedelta(hours=5)
    today    = est_now.strftime("%Y-%m-%d")
    filename = f"{args.stream}_latest.json"

    try:
        from huggingface_hub import hf_hub_download
        import json

        local = hf_hub_download(
            repo_id="P2SAMAPA/p2-etf-vlstm-outputs",
            filename=filename,
            repo_type="dataset",
            token=hf_token,
            force_download=True,
        )
        with open(local) as f:
            data = json.load(f)

        last_run = data.get("run_date", "")

        if last_run == today:
            print(f"⏭️  {args.stream} stream already ran today ({today}). Skipping.")
            sys.exit(0)   # already ran — workflow will skip
        else:
            print(f"✅ Last run: {last_run}. Today: {today}. Proceeding.")
            sys.exit(1)   # not yet run — proceed

    except Exception as e:
        # File doesn't exist yet (first ever run) or any other error → proceed
        print(f"ℹ️  Could not read {filename} ({e}). Proceeding with training.")
        sys.exit(1)   # proceed


if __name__ == "__main__":
    main()
