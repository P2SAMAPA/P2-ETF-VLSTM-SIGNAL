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

try:
    from config import get_config
except ImportError:
    def get_config(universe):
        return {
            "output_dataset": "P2SAMAPA/p2-etf-vlstm-outputs",
            "file_prefix": "fi",
        }


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

    config       = get_config(args.universe)
    dataset_name = config.get("output_dataset", "P2SAMAPA/p2-etf-vlstm-outputs")
    file_prefix  = config.get("file_prefix", args.universe)   # FIX: was missing entirely

    est_now  = datetime.now(timezone.utc) - timedelta(hours=5)
    today    = est_now.strftime("%Y-%m-%d")

    # FIX: was f"{args.stream}_latest.json" — missing the prefix, so it was
    # looking for "expanding_latest.json" instead of "fi_expanding_latest.json".
    # That caused an exception → always printed PROCEED (harmless but wrong).
    # Now correctly matches what writer.py actually uploads.
    filename = f"{file_prefix}_{args.stream}_latest.json"

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
        # File doesn't exist yet or any error → always proceed
        print("PROCEED")


if __name__ == "__main__":
    main()
