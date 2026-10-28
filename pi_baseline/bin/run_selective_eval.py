#!/usr/bin/env python3
"""
Selective Evaluation CLI for Isolated Pi Baseline.

Enables executing individual tasks or subsets of evaluation seeds
against arm_0_no_memory and/or arm_1_ordinary_rag without running
the entire 100-run calibration matrix.

Usage Examples:
  # 1. Run a single coding task on Arm 0 with seed 42
  python3 research/pi_baseline/bin/run_selective_eval.py --arm arm_0_no_memory --tasks code_01_binary_search_partition --seeds 42

  # 2. Run two tasks on both Arm 0 and Arm 1 across seeds 42 and 137
  python3 research/pi_baseline/bin/run_selective_eval.py --arm both --tasks code_01_binary_search_partition,ir_01_dependency_maintainer --seeds 42,137

  # 3. List all available benchmark tasks
  python3 research/pi_baseline/bin/run_selective_eval.py --list-tasks
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Ensure harness is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_DIR = SCRIPT_DIR.parent
HARNESS_DIR = BASELINE_DIR / "harness"
sys.path.insert(0, str(HARNESS_DIR))

from arm0_runner import Arm0Runner
from arm1_runner import Arm1Runner


def load_manifest() -> Dict[str, Any]:
    manifest_path = BASELINE_DIR / "tasks" / "calibration_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_available_tasks(manifest: Dict[str, Any]):
    print("\nAvailable Benchmark Tasks:")
    print("--------------------------------------------------------------------------------")
    print(f"{'TYPE':<8} | {'TASK ID':<35} | {'TITLE'}")
    print("--------------------------------------------------------------------------------")
    for t in manifest.get("coding_tasks", []):
        print(f"Coding   | {t['task_id']:<35} | {t['title']}")
    for t in manifest.get("ir_tasks", []):
        print(f"Multi-IR | {t['task_id']:<35} | {t['title']}")
    print("--------------------------------------------------------------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Run selective live or offline benchmark tasks on Arm 0 and Arm 1.")
    parser.add_argument("--arm", choices=["arm_0_no_memory", "arm_1_ordinary_rag", "both"], default="arm_0_no_memory",
                        help="Experimental arm to execute (default: arm_0_no_memory)")
    parser.add_argument("--tasks", type=str, default="",
                        help="Comma-separated list of task IDs to run (e.g. code_01_binary_search_partition,ir_01_dependency_maintainer)")
    parser.add_argument("--type", choices=["all", "coding", "ir"], default="all",
                        help="Filter tasks by domain if --tasks is not specified")
    parser.add_argument("--seeds", type=str, default="42",
                        help="Comma-separated list of evaluation seeds to run (e.g. 42 or 42,137, default: 42)")
    parser.add_argument("--list-tasks", action="store_true",
                        help="List all available benchmark tasks and exit")

    args = parser.parse_args()
    manifest = load_manifest()

    if args.list_tasks:
        list_available_tasks(manifest)
        sys.exit(0)

    # 1. Parse seeds
    try:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    except ValueError:
        print(f"Error: Invalid seeds '{args.seeds}'. Must be integers.", file=sys.stderr)
        sys.exit(1)

    # 2. Collect target tasks
    all_tasks = manifest.get("coding_tasks", []) + manifest.get("ir_tasks", [])
    task_map = {t["task_id"]: t for t in all_tasks}

    if args.tasks:
        requested_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
        selected_tasks = []
        for tid in requested_ids:
            if tid in task_map:
                selected_tasks.append(task_map[tid])
            else:
                print(f"Warning: Task ID '{tid}' not found in manifest. Skipping.", file=sys.stderr)
    else:
        if args.type == "coding":
            selected_tasks = manifest.get("coding_tasks", [])
        elif args.type == "ir":
            selected_tasks = manifest.get("ir_tasks", [])
        else:
            selected_tasks = all_tasks

    if not selected_tasks:
        print("No valid tasks selected. Use --list-tasks to see available options.", file=sys.stderr)
        sys.exit(1)

    # 3. Detect API Key Status
    api_key_present = bool(
        os.environ.get("PARTICLE_API_KEY")
        or os.environ.get("ZHIPU_API_KEY")
        or os.environ.get("BIGMODEL_API_KEY")
    )
    mode_str = "LIVE API (Particle / GLM 5.3 Flash)" if api_key_present else "OFFLINE CALIBRATED SIMULATION"

    print("\n================================================================================")
    print("SELECTIVE EVALUATION RUNNER")
    print(f"Mode:          {mode_str}")
    print(f"Arms:          {args.arm}")
    print(f"Tasks ({len(selected_tasks)}):    {[t['task_id'] for t in selected_tasks]}")
    print(f"Seeds ({len(seeds)}):    {seeds}")
    print("================================================================================\n")

    # 4. Instantiate runners
    config_file = str(BASELINE_DIR / "config" / "model_config.json")
    runners = {}
    if args.arm in ["arm_0_no_memory", "both"]:
        runners["arm_0_no_memory"] = Arm0Runner(config_file)
    if args.arm in ["arm_1_ordinary_rag", "both"]:
        runners["arm_1_ordinary_rag"] = Arm1Runner(config_file)

    results = []
    total_runs = len(runners) * len(selected_tasks) * len(seeds)
    run_idx = 0

    for arm_id, runner in runners.items():
        print(f"\n>>> Executing Arm: {arm_id} <<<")
        for task in selected_tasks:
            task_id = task["task_id"]
            is_coding = "initial_code" in task

            for seed in seeds:
                run_idx += 1
                prefix = f"[{run_idx}/{total_runs}] {arm_id} | {task_id} | Seed {seed}"
                print(f"{prefix:<60} ... ", end="", flush=True)

                t_start = time.perf_counter()
                if is_coding:
                    res = runner.execute_coding_task(task, seed)
                else:
                    res = runner.execute_ir_task(task, seed)
                elapsed = (time.perf_counter() - t_start) * 1000.0

                passed = res.get("passed", False)
                status_str = "PASS" if passed else "FAIL"
                tokens = res.get("total_tokens", 0)
                cost = res.get("cost_usd", 0.0)

                print(f"{status_str} (Tokens: {tokens}, Latency: {elapsed:.1f}ms, Cost: ${cost:.6f})")
                results.append({
                    "arm": arm_id,
                    "task_id": task_id,
                    "seed": seed,
                    "passed": passed,
                    "tokens": tokens,
                    "latency_ms": elapsed,
                    "cost_usd": cost,
                    "run_dir": str(res.get("run_dir", ""))
                })

    # 5. Summary Table
    print("\n================================================================================")
    print("EXECUTION SUMMARY")
    print("--------------------------------------------------------------------------------")
    print(f"{'ARM':<18} | {'TASK ID':<32} | {'SEED':<5} | {'STATUS':<6} | {'TOKENS':<7} | {'LATENCY'}")
    print("--------------------------------------------------------------------------------")
    pass_count = 0
    for r in results:
        status_label = "PASS" if r["passed"] else "FAIL"
        if r["passed"]:
            pass_count += 1
        print(f"{r['arm']:<18} | {r['task_id']:<32} | {r['seed']:<5} | {status_label:<6} | {r['tokens']:<7} | {r['latency_ms']:.1f}ms")
    print("--------------------------------------------------------------------------------")
    pass_rate = (pass_count / len(results) * 100.0) if results else 0.0
    print(f"Total Completed: {len(results)} | Passed: {pass_count}/{len(results)} ({pass_rate:.1f}%)\n")


if __name__ == "__main__":
    main()
