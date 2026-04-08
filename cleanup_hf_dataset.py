#!/usr/bin/env python3
"""
cleanup_hf_dataset.py

Script to clean up corrupted HF dataset with mixed FI/Equity schemas.
Run this locally to fix the dataset before deploying the corrected code.
"""

import os
import sys
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
import io

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("❌ Set HF_TOKEN environment variable")
    sys.exit(1)

DATASET = "P2SAMAPA/p2-etf-vlstm-outputs"
api = HfApi(token=HF_TOKEN)

print("🔍 Checking current dataset files...")
files = api.list_repo_files(DATASET, repo_type="dataset")
print(f"Found files: {files}")

# Files to delete (corrupted mixed-schema files)
files_to_delete = [
    "history.parquet",  # Old mixed-schema file
    "latest.json",      # Old non-prefixed file
    "expanding_latest.json",
    "shrinking_latest.json",
]

print("\n🗑️ Cleaning up corrupted files...")
for filename in files_to_delete:
    if filename in files:
        try:
            api.delete_file(
                path_in_repo=filename,
                repo_id=DATASET,
                repo_type="dataset",
                commit_message=f"Cleanup: Remove corrupted {filename}"
            )
            print(f"  ✅ Deleted {filename}")
        except Exception as e:
            print(f"  ⚠️ Could not delete {filename}: {e}")

# Check for existing prefixed files
prefixed_files = [f for f in files if f.startswith(("fi_", "equity_"))]
print(f"\n📁 Existing prefixed files: {prefixed_files}")

# Initialize empty history files for both universes if they don't exist
for prefix in ["fi", "equity"]:
    history_file = f"{prefix}_history.parquet"
    if history_file not in files:
        print(f"\n📄 Creating empty {history_file}...")
        df = pd.DataFrame(columns=[
            "run_date", "data_through",
            "exp_signal", "exp_strength", "exp_score_pct", "exp_agreement",
            "shr_signal", "shr_strength", "shr_score_pct", "shr_agreement",
            "universe"
        ])
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        api.upload_file(
            path_or_fileobj=buf,
            path_in_repo=history_file,
            repo_id=DATASET,
            repo_type="dataset",
            commit_message=f"Init: Create empty {history_file}"
        )
        print(f"  ✅ Created {history_file}")

print("\n✅ Cleanup complete!")
print("\nNext steps:")
print("1. Deploy the corrected code files")
print("2. Run the workflows manually to test")
print("3. Update your Streamlit app to handle the new file prefixes")
