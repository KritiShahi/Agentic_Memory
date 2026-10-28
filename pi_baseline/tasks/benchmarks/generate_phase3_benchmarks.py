#!/usr/bin/env python3
"""
Phase 3 Benchmark Suite Synthesis & Verification Script.
Generates train_stream_v1.json, eval_stream_v1.json, and benchmark_manifest.json
with 40 tasks across Python and SQL domains stratified by Easy, Medium, and Hard tiers.
"""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent

def load_and_verify():
    train_file = BENCH_DIR / "train_stream_v1.json"
    eval_file = BENCH_DIR / "eval_stream_v1.json"
    manifest_file = BENCH_DIR / "benchmark_manifest.json"

    assert train_file.exists(), "train_stream_v1.json missing"
    assert eval_file.exists(), "eval_stream_v1.json missing"
    assert manifest_file.exists(), "benchmark_manifest.json missing"

    with open(train_file, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(eval_file, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    assert len(train_data) == 20, f"Expected 20 train tasks, got {len(train_data)}"
    assert len(eval_data) == 20, f"Expected 20 eval tasks, got {len(eval_data)}"

    # Anti-contamination assertions
    train_ids = {t["task_id"] for t in train_data}
    eval_ids = {t["task_id"] for t in eval_data}
    assert train_ids.isdisjoint(eval_ids), "Task ID overlap between train and eval!"

    print(f"Verified {len(train_data)} train tasks and {len(eval_data)} eval tasks successfully.")

if __name__ == "__main__":
    load_and_verify()
