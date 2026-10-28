#!/usr/bin/env python3
"""
Example 3: End-to-End Task Execution & Test Assertion Check.

This script demonstrates:
1. Loading a real benchmark coding task from tasks/calibration_manifest.json.
2. Executing the task using Arm0Runner (Stateless baseline).
3. Running the automated test harness against the model-generated code in an isolated directory.
4. Verifying test pass/fail status, token consumption, latency, and saved run artifacts.
"""

import json
import os
import sys
from pathlib import Path

# Paths relative to this script
EXAMPLES_DIR = Path(__file__).resolve().parent
BASELINE_DIR = EXAMPLES_DIR.parent
HARNESS_DIR = BASELINE_DIR / "harness"
sys.path.insert(0, str(HARNESS_DIR))

from arm0_runner import Arm0Runner

def main():
    print("==================================================================")
    print("EXAMPLE 3: END-TO-END CODING TASK EXECUTION & TEST CHECK")
    print("==================================================================")

    # 1. Load benchmark task manifest
    manifest_file = BASELINE_DIR / "tasks" / "calibration_manifest.json"
    print(f"\n1. Loading benchmark task from: {manifest_file.relative_to(BASELINE_DIR)}")
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    task = manifest["coding_tasks"][0]
    task_id = task["task_id"]
    print(f"   - Task ID:    {task_id}")
    print(f"   - Title:      {task['title']}")
    print(f"   - Problem:\n     {task['instruction']}")

    # 2. Instantiate Arm0Runner
    config_path = BASELINE_DIR / "config" / "model_config.json"
    runner = Arm0Runner(str(config_path))
    
    # 3. Execute Coding Task with Seed 42
    seed = 42
    print(f"\n2. Executing task on Arm 0 (No Memory) with Seed {seed}...")
    res = runner.execute_coding_task(task, seed=seed)

    # 4. Display Execution Results & Metrics
    passed = res.get("passed", False)
    status_str = "PASS (Exit Code 0)" if passed else f"FAIL (Exit Code {res.get('exit_code', 1)})"
    
    print(f"\n3. Execution Telemetry Results:")
    print(f"   - Test Status:       {status_str}")
    print(f"   - Prompt Tokens:     {res.get('prompt_tokens', 0)}")
    print(f"   - Completion Tokens: {res.get('completion_tokens', 0)}")
    print(f"   - Total Tokens:      {res.get('total_tokens', 0)}")
    print(f"   - Latency:           {res.get('latency_ms', 0):.2f} ms")
    print(f"   - Estimated Cost:    ${res.get('cost_usd', 0.0):.6f}")

    # 5. Check Permanent Run Artifacts Directory
    run_dir = Path(res.get("run_dir", BASELINE_DIR / "runs" / "arm_0_no_memory" / f"{task_id}_seed_{seed}"))
    print(f"\n4. Checking saved execution artifacts in:")
    print(f"   {run_dir.relative_to(BASELINE_DIR) if run_dir.is_relative_to(BASELINE_DIR) else run_dir}")
    
    assert (run_dir / "input" / "task_spec.json").exists(), "Missing input/task_spec.json"
    assert (run_dir / "execution").exists(), "Missing execution directory"
    assert (run_dir / "output" / "summary.json").exists(), "Missing output/summary.json"
    print("   [PASS] All input files, execution code, and output summaries permanently stored.")

    print("\n==================================================================")
    print("RESULT: End-to-End Task Evaluation Pipeline Verified Accurately!")
    print("==================================================================\n")

if __name__ == "__main__":
    main()
