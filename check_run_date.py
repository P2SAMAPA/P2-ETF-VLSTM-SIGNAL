"""
check_run_date.py
P2-ETF-VLSTM-SIGNAL

Checks if this stream already ran today (EST) for a specific universe.
Always exits 0 — result communicated via stdout for the yml to capture.

Prints either:
  ALREADY_RAN_TODAY
  PROCEED

Usage:
  python check_run_date.py --stream expanding --universe fi
  python check_run_date.py --stream shrinking --universe equity
"""

import sys
import argparse
import os
from datetime import datetime, timezone, timedelta

# Import config to get dataset names per universe
try:
    from config import get_config
except ImportError:
    # Fallback if config.py doesn't exist (old code) – use default dataset
    def get_config(universe):
        return {"output_dataset": "P2SAMAPA/p2-etf-vlstm-outputs"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", required=True,
                        choices=["expanding", "shrinking"])
    parser.add_argument("--universe", default="fi",
                        choices=["fi", "equity"],
                        help="ETF universe (default: fi)")
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        print("PROCEED")
        return

    # Get dataset name for this universe
    config = get_config(args.universe)
    dataset_name = config.get("output_dataset", "P2SAMAPA/p2-etf-vlstm-outputs")

    est_now  = datetime.now(timezone.utc) - timedelta(hours=5)
    today    = est_now.strftime("%Y-%m-%d")
    filename = f"{args.stream}_latest.json"

    try:
        from huggingface_hub import hf_hub_download
        import json

        local = hf_hub_download(
            repo_id=dataset_name,
            filename=filename,
            repo_type="dataset",
            token=hf_token,
            force_download=True,
        )
        with open(local) as f:
            data = json.load(f)

        last_run = data.get("run_date", "")

        if last_run == today:
            print("ALREADY_RAN_TODAY")
        else:
            print("PROCEED")

    except Exception:
        # File doesn't exist yet or any error → proceed
        print("PROCEED")


if __name__ == "__main__":
    main()
