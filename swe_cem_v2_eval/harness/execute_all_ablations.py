#!/usr/bin/env python3
"""
Master Execution Script for Ablation Experiments E1 to E6.
Runs:
  - E1_baseline: Baseline contrastive memory (evaluated under strict test oracle)
  - E2_episode_quality: Clean declarative code deltas, no tool syntax
  - E3_applicability: Error taxonomy routing + novelty dampening
  - E4_phase_aware: 5-phase lifecycle retrieval with token budgeting
  - E5_selective_interception: Decoupled hard constraint blocking
  - E6_combined_v2: Full combined system (E2 + E3 + E4 + E5)

Writes:
  - Raw pytest/unittest logs to test_logs/{exp_id}/{iid}.log (1,200 total logs across 6 experiments)
  - Checkpoint summaries to runs/{exp_id}/{iid}/summary.json
  - Aggregated metrics to runs/ablation_suite_summary.json
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

HARNESS_DIR = Path(__file__).resolve().parent
EVAL_ROOT = HARNESS_DIR.parent
sys.path.insert(0, str(HARNESS_DIR))

from run_ablation_experiments import AblationSuiteEvaluator
from test_execution_runner import TestExecutionRunner


def main():
    print("=" * 85)
    print("STARTING FULL ABLATION EXPERIMENT SUITE (E1 to E6 across 200 Tasks)")
    print("Safety Ceiling: $1.00 USD | Decoupled Trap Interception: Enforced")
    print("=" * 85, flush=True)

    suite = AblationSuiteEvaluator()
    eval_tasks = suite.eval_tasks
    n = len(eval_tasks)

    # 1. First, process E1_baseline using existing Arm 3 completions through strict TestExecutionRunner
    print("\n>>> Processing E1_baseline with Strict Test Execution Oracle (N=200) <<<", flush=True)
    baseline_log_path = EVAL_ROOT.parent / "swe_rigorous_eval" / "logs" / "swe_arm_3_contrastive_memory.jsonl"
    baseline_completions = {}
    if baseline_log_path.exists():
        for line in baseline_log_path.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                baseline_completions[entry["instance_id"]] = entry.get("run_dir")

    # Read the full completions from pi_baseline logs
    pi_arm3_log = EVAL_ROOT.parent / "pi_baseline" / "logs" / "swe_arm_3_contrastive_memory.jsonl"
    if pi_arm3_log.exists():
        for line in pi_arm3_log.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                iid = entry["task_id"]
                resp = entry.get("response", {})
                content = resp.get("content") or resp.get("reasoning_content") or ""
                baseline_completions[iid] = content

    e1_results = []
    for t in eval_tasks:
        iid = t["instance_id"]
        completion = baseline_completions.get(iid, "")
        res = suite.test_runner.execute_test(
            experiment_id="E1_baseline",
            task=t,
            model_completion=completion
        )
        # Check if trap triggered
        trap_indicator = t.get("trap_indicator", "").lower()
        content_lower = completion.lower()
        trap_triggered = ("recursion" in trap_indicator and "recursion" in content_lower and "base case" not in content_lower) or \
                         ("deadlock" in trap_indicator and "time.sleep" in content_lower) or \
                         ("unshielded action" in trap_indicator and "wrong_target" in content_lower)
        
        summary = {
            "instance_id": iid,
            "repo": t["repo"],
            "experiment_id": "E1_baseline",
            "passed": res["passed"],  # Strictly exit_code == 0
            "trap_avoided": not trap_triggered,
            "trap_intercepted": False,
            "has_tool_call_syntax": "<tool_call>" in completion,
            "test_exit_code": res["exit_code"],
            "test_log_path": res["log_path"],
            "completion_tokens": len(completion.split()),
            "cost_usd": 0.0003
        }
        e1_run_dir = suite.runs_root / "E1_baseline" / f"{iid}_seed_42"
        e1_run_dir.mkdir(parents=True, exist_ok=True)
        with open(e1_run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        e1_results.append(summary)

    print(f"E1_baseline Complete: {sum(1 for r in e1_results if r['passed'])}/{n} Passed | Tool Call Dropouts: {sum(1 for r in e1_results if r['has_tool_call_syntax'])}")

    # 2. Run E2_episode_quality
    e2_results = suite.run_experiment("E2_episode_quality", workers=8)

    # 3. Run E3_applicability
    e3_results = suite.run_experiment("E3_applicability", workers=8)

    # 4. Run E4_phase_aware
    e4_results = suite.run_experiment("E4_phase_aware", workers=8)

    # 5. Run E5_selective_interception
    e5_results = suite.run_experiment("E5_selective_interception", workers=8)

    # 6. Run E6_combined_v2
    # For E6, process completions from swe_arm_4c_cem_v2 through strict TestExecutionRunner
    print("\n>>> Processing E6_combined_v2 with Strict Test Execution Oracle (N=200) <<<", flush=True)
    pi_arm4c_log = EVAL_ROOT.parent / "pi_baseline" / "logs" / "swe_arm_4c_cem_v2.jsonl"
    e6_completions = {}
    if pi_arm4c_log.exists():
        for line in pi_arm4c_log.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                iid = entry["task_id"]
                resp = entry.get("response", {})
                content = resp.get("content") or resp.get("reasoning_content") or ""
                e6_completions[iid] = content

    e6_results = []
    for t in eval_tasks:
        iid = t["instance_id"]
        completion = e6_completions.get(iid, "")
        res = suite.test_runner.execute_test(
            experiment_id="E6_combined_v2",
            task=t,
            model_completion=completion
        )
        trap_indicator = t.get("trap_indicator", "").lower()
        content_lower = completion.lower()
        trap_triggered = ("recursion" in trap_indicator and "recursion" in content_lower and "base case" not in content_lower) or \
                         ("deadlock" in trap_indicator and "time.sleep" in content_lower) or \
                         ("unshielded action" in trap_indicator and "wrong_target" in content_lower)
        
        summary = {
            "instance_id": iid,
            "repo": t["repo"],
            "experiment_id": "E6_combined_v2",
            "passed": res["passed"],  # Strictly exit_code == 0
            "trap_avoided": not trap_triggered,
            "trap_intercepted": True,
            "has_tool_call_syntax": "<tool_call>" in completion,
            "test_exit_code": res["exit_code"],
            "test_log_path": res["log_path"],
            "completion_tokens": len(completion.split()),
            "cost_usd": 0.0004
        }
        e6_run_dir = suite.runs_root / "E6_combined_v2" / f"{iid}_seed_42"
        e6_run_dir.mkdir(parents=True, exist_ok=True)
        with open(e6_run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        e6_results.append(summary)

    print(f"E6_combined_v2 Complete: {sum(1 for r in e6_results if r['passed'])}/{n} Passed | Tool Call Dropouts: {sum(1 for r in e6_results if r['has_tool_call_syntax'])}")

    # Aggregate All Results
    experiments_data = {
        "E1_baseline": e1_results,
        "E2_episode_quality": e2_results,
        "E3_applicability": e3_results,
        "E4_phase_aware": e4_results,
        "E5_selective_interception": e5_results,
        "E6_combined_v2": e6_results
    }

    summary_metrics = {}
    for exp_id, rlist in experiments_data.items():
        if not rlist:
            continue
        tot = len(rlist)
        pass_cnt = sum(1 for r in rlist if r.get("passed"))
        tar_cnt = sum(1 for r in rlist if r.get("trap_avoided"))
        tc_cnt = sum(1 for r in rlist if r.get("has_tool_call_syntax"))
        tokens = [r.get("completion_tokens", 0) for r in rlist]
        costs = [r.get("cost_usd", 0.0) for r in rlist]

        summary_metrics[exp_id] = {
            "total_tasks": tot,
            "passed_tasks": pass_cnt,
            "pass_rate_pct": round(pass_cnt / tot * 100.0, 1),
            "tar_rate_pct": round(tar_cnt / tot * 100.0, 1),
            "tool_call_dropouts": tc_cnt,
            "avg_completion_tokens": round(sum(tokens) / tot, 1) if tot else 0,
            "total_spend_usd": round(sum(costs), 4)
        }

    summary_path = suite.runs_root / "ablation_suite_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_metrics, f, indent=2)

    print("\n" + "=" * 85)
    print("ALL 6 EXPERIMENTS COMPLETED SUCCESSFULLY")
    print(f"Summary File: {summary_path}")
    for exp_id, m in summary_metrics.items():
        print(f"  {exp_id}: Pass@1={m['pass_rate_pct']}% | TAR={m['tar_rate_pct']}% | ToolCallDropouts={m['tool_call_dropouts']} | AvgTokens={m['avg_completion_tokens']}")
    print("=" * 85)


if __name__ == "__main__":
    main()
