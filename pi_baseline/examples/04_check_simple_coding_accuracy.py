#!/usr/bin/env python3
"""
Example 4: Simple Coding Tasks Accuracy, Telemetry Logging & Disk Artifact Check.

This script tests:
  1. Task 1: Find Even Numbers (code_06_find_even_numbers)
  2. Task 2: Binary Search (code_07_binary_search_index)

Ensures that:
  - Every model call writes structured telemetry to logs/arm_0_no_memory.jsonl.
  - Every task saves permanent execution artifacts to runs/arm_0_no_memory/<task_id>_seed_42/.
  - Unit test assertions validate the generated code dynamically.
  - Disk logs are verified and printed back to confirm persistence.
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

# Paths relative to this script
EXAMPLES_DIR = Path(__file__).resolve().parent
BASELINE_DIR = EXAMPLES_DIR.parent
HARNESS_DIR = BASELINE_DIR / "harness"
sys.path.insert(0, str(HARNESS_DIR))

from glm_provider_client import GLM53FlashClient
from arm0_runner import Arm0Runner


def main():
    print("================================================================================")
    print("EXAMPLE 4: EVALUATING SIMPLE CODING ACCURACY WITH FULL TELEMETRY PERSISTENCE")
    print("================================================================================")

    config_path = BASELINE_DIR / "config" / "model_config.json"
    manifest_path = BASELINE_DIR / "tasks" / "calibration_manifest.json"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    task_map = {t["task_id"]: t for t in manifest.get("coding_tasks", [])}
    
    runner = Arm0Runner(str(config_path))
    api_key_set = bool(runner.client.api_key)
    mode_str = "LIVE API (Particle / GLM 5.3 Flash)" if api_key_set else "OFFLINE CALIBRATED SIMULATION"
    print(f"Model: {runner.client.model} | Endpoint: {runner.client.endpoint_url} | Mode: {mode_str}\n")

    tasks_to_test = ["code_06_find_even_numbers", "code_07_binary_search_index"]
    seed = 42

    initial_log_size = (BASELINE_DIR / "logs" / "arm_0_no_memory.jsonl").stat().st_size if (BASELINE_DIR / "logs" / "arm_0_no_memory.jsonl").exists() else 0

    results = []

    for idx, task_id in enumerate(tasks_to_test, 1):
        task = task_map.get(task_id)
        assert task is not None, f"Task {task_id} not found in calibration manifest"

        print("--------------------------------------------------------------------------------")
        print(f"TASK {idx}: {task['title']} (`{task_id}`)")
        print("--------------------------------------------------------------------------------")
        print(f"Instruction: {task['instruction']}")

        # 1. Execute task via Arm0Runner (performs inference, creates run_dir, runs unit tests, logs telemetry)
        t_start = time.perf_counter()
        exec_res = runner.execute_coding_task(task, seed=seed)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        passed = exec_res.get("passed", False)
        status_label = "PASS (100% Tests Green)" if passed else "FAIL"

        print(f"\nExecution Outcome:")
        print(f"  - Status:            {status_label}")
        print(f"  - Exit Code:         {exec_res.get('exit_code')}")
        print(f"  - Prompt Tokens:     {exec_res.get('prompt_tokens')}")
        print(f"  - Completion Tokens: {exec_res.get('completion_tokens')}")
        print(f"  - Latency:           {elapsed_ms:.2f} ms")
        print(f"  - Step Cost:         ${exec_res.get('cost_usd', 0.0):.6f}")

        run_dir = Path(exec_res["run_dir"])
        print(f"  - Saved Run Dir:     {run_dir.relative_to(BASELINE_DIR)}")

        # Verify artifacts in run_dir
        assert (run_dir / "input" / "task_spec.json").exists()
        assert (run_dir / "output" / "summary.json").exists()
        assert (run_dir / "output" / "stdout.log").exists()

        results.append({
            "task_id": task_id,
            "title": task["title"],
            "passed": passed,
            "tokens": exec_res.get("total_tokens"),
            "run_dir": run_dir
        })
        print()

    # -------------------------------------------------------------------------
    # 2. VERIFY TELEMETRY DISK PERSISTENCE
    # -------------------------------------------------------------------------
    print("================================================================================")
    print("VERIFYING TELEMETRY DISK LOGS (logs/arm_0_no_memory.jsonl)")
    print("================================================================================")
    
    log_file = BASELINE_DIR / "logs" / "arm_0_no_memory.jsonl"
    assert log_file.exists(), "Log file does not exist"
    
    final_log_size = log_file.stat().st_size
    print(f"Telemetry log file: {log_file.relative_to(BASELINE_DIR)}")
    print(f"Initial Size: {initial_log_size} bytes -> Final Size: {final_log_size} bytes (+{final_log_size - initial_log_size} bytes appended)")

    # Read last N lines corresponding to our runs
    with open(log_file, "r", encoding="utf-8") as f:
        all_lines = [l.strip() for l in f if l.strip()]

    # Extract log entries for our tasks
    matching_entries = []
    for line in all_lines:
        try:
            entry = json.loads(line)
            if entry.get("task_id") in tasks_to_test:
                matching_entries.append(entry)
        except json.JSONDecodeError:
            pass

    print(f"\nFound {len(matching_entries)} persisted telemetry entries for simple coding tasks:\n")
    for entry in matching_entries[-2:]:
        print(f"• Task: {entry['task_id']} | Seed: {entry['seed']} | Tokens: {entry['total_tokens']} | Timestamp: {entry['timestamp']}")
        print(f"  Response Content: {entry['response']['content'][:100]}...\n")

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("================================================================================")
    print("FINAL SUMMARY")
    print("================================================================================")
    for r in results:
        v = "PASS" if r["passed"] else "FAIL"
        print(f"{r['task_id']:<32} | {r['title']:<40} | {v}")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
