"""
check_run_date.py
P2-ETF-VLSTM-SIGNAL

Checks if this stream already ran today (EST).
Always exits 0 — result communicated via stdout for the yml to capture.

Prints either:
  ALREADY_RAN_TODAY
  PROCEED

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
        print("PROCEED")
        return

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
            print(f"ALREADY_RAN_TODAY")
        else:
            print(f"PROCEED")

    except Exception:
        # File doesn't exist yet or any error → proceed
        print("PROCEED")


if __name__ == "__main__":
    main()
