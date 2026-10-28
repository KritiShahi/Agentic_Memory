#!/usr/bin/env python3
"""
Preliminary 3-Arm Comparative Baseline Test.
Evaluates Arm 0 (No Memory), Arm 1 (Ordinary RAG), and Arm 3 (Contrastive Memory + Intercept)
on the 6-task held-out evaluation probe (Easy, Medium, Hard across Python & SQL).
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Setup paths
HARNESS_DIR = Path(__file__).resolve().parent
BASELINE_DIR = HARNESS_DIR.parent
sys.path.insert(0, str(HARNESS_DIR))
sys.path.insert(0, str(HARNESS_DIR / "memory"))
sys.path.insert(0, str(HARNESS_DIR / "hooks"))

from benchmark_loader import BenchmarkLoader
from arm0_runner import Arm0Runner
from arm1_runner import Arm1Runner
from arm3_runner import Arm3Runner


def run_preliminary_comparison():
    loader = BenchmarkLoader()
    probe_tasks = loader.get_representative_subset(per_tier=1)
    
    print("=" * 80)
    print(f"PRELIMINARY 3-ARM COMPARATIVE BASELINE EVALUATION (N={len(probe_tasks)} Tasks)")
    print("Arms: Arm 0 (No Memory) vs Arm 1 (Ordinary RAG) vs Arm 3 (Contrastive Memory + Intercept)")
    print("=" * 80)
    for t in probe_tasks:
        print(f"  • [{t['domain'].upper():6s} | {t['complexity'].upper():6s}] {t['task_id']}: {t['title']}")
    print("-" * 80)

    # Initialize runners
    runners = {
        "arm_0_no_memory": Arm0Runner(),
        "arm_1_ordinary_rag": Arm1Runner(),
        "arm_3_contrastive_memory": Arm3Runner()
    }

    results_by_arm: Dict[str, List[Dict[str, Any]]] = {k: [] for k in runners}
    eval_seed = 42

    for arm_id, runner in runners.items():
        print(f"\n>>> Executing Evaluation for [{arm_id.upper()}] (Seed={eval_seed}) <<<")
        for idx, task in enumerate(probe_tasks, 1):
            t_id = task["task_id"]
            print(f"  [{idx}/{len(probe_tasks)}] Running {t_id}...", end=" ", flush=True)
            res = runner.execute_coding_task(task, seed=eval_seed)
            results_by_arm[arm_id].append(res)
            status_str = "PASS" if res.get("passed") else "FAIL"
            print(f"[{status_str}] (Latency: {res.get('latency_ms', 0):.1f}ms | Tokens: {res.get('total_tokens', 0)})")

    # Aggregate Metrics
    summary_metrics = {}
    for arm_id, run_list in results_by_arm.items():
        total = len(run_list)
        passes = sum(1 for r in run_list if r.get("passed"))
        pass_rate = (passes / total) * 100.0 if total else 0.0

        # Trap Avoidance Rate (TAR)
        # Arm 3 explicitly tracks trap_avoided; for baseline arms, pass without failure equals trap avoided
        if arm_id == "arm_3_contrastive_memory":
            tar_count = sum(1 for r in run_list if r.get("trap_avoided", r.get("passed")))
        else:
            tar_count = passes
        tar_rate = (tar_count / total) * 100.0 if total else 0.0

        avg_prompt_tokens = sum(r.get("prompt_tokens", 0) for r in run_list) / total if total else 0
        avg_memory_tokens = sum(r.get("memory_injected_tokens", 0) for r in run_list) / total if total else 0
        avg_completion_tokens = sum(r.get("completion_tokens", 0) for r in run_list) / total if total else 0
        avg_total_tokens = sum(r.get("total_tokens", 0) for r in run_list) / total if total else 0
        avg_latency_ms = sum(r.get("latency_ms", 0) for r in run_list) / total if total else 0
        total_cost_usd = sum(r.get("cost_usd", 0) for r in run_list)

        summary_metrics[arm_id] = {
            "total_tasks": total,
            "passed_tasks": passes,
            "pass_rate_pct": pass_rate,
            "trap_avoidance_rate_pct": tar_rate,
            "turn_1_success_pct": pass_rate,
            "avg_prompt_tokens": round(avg_prompt_tokens, 1),
            "avg_memory_tokens": round(avg_memory_tokens, 1),
            "avg_completion_tokens": round(avg_completion_tokens, 1),
            "avg_total_tokens": round(avg_total_tokens, 1),
            "avg_latency_ms": round(avg_latency_ms, 1),
            "total_cost_usd": round(total_cost_usd, 6),
            "task_breakdown": [
                {
                    "task_id": r["task_id"],
                    "passed": r["passed"],
                    "exit_code": r["exit_code"],
                    "latency_ms": round(r["latency_ms"], 1)
                }
                for r in run_list
            ]
        }

    # Save summary artifact
    runs_dir = BASELINE_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_file = runs_dir / "preliminary_comparison_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    # Print Summary Table
    print("\n" + "=" * 90)
    print("PRELIMINARY COMPARATIVE BASELINE EVALUATION RESULTS")
    print("=" * 90)
    print(f"{'Evaluation Metric':<28} | {'Arm 0 (No Memory)':<18} | {'Arm 1 (Flat RAG)':<18} | {'Arm 3 (Contrastive)':<18}")
    print("-" * 90)
    print(f"{'Pass@1 Success Rate':<28} | {summary_metrics['arm_0_no_memory']['pass_rate_pct']:>16.1f}% | {summary_metrics['arm_1_ordinary_rag']['pass_rate_pct']:>16.1f}% | {summary_metrics['arm_3_contrastive_memory']['pass_rate_pct']:>16.1f}%")
    print(f"{'Trap Avoidance Rate (TAR)':<28} | {summary_metrics['arm_0_no_memory']['trap_avoidance_rate_pct']:>16.1f}% | {summary_metrics['arm_1_ordinary_rag']['trap_avoidance_rate_pct']:>16.1f}% | {summary_metrics['arm_3_contrastive_memory']['trap_avoidance_rate_pct']:>16.1f}%")
    print(f"{'Turn-1 Success Rate':<28} | {summary_metrics['arm_0_no_memory']['turn_1_success_pct']:>16.1f}% | {summary_metrics['arm_1_ordinary_rag']['turn_1_success_pct']:>16.1f}% | {summary_metrics['arm_3_contrastive_memory']['turn_1_success_pct']:>16.1f}%")
    print(f"{'Avg Memory Tokens':<28} | {summary_metrics['arm_0_no_memory']['avg_memory_tokens']:>16.1f}  | {summary_metrics['arm_1_ordinary_rag']['avg_memory_tokens']:>16.1f}  | {summary_metrics['arm_3_contrastive_memory']['avg_memory_tokens']:>16.1f} ")
    print(f"{'Avg Total Tokens':<28} | {summary_metrics['arm_0_no_memory']['avg_total_tokens']:>16.1f}  | {summary_metrics['arm_1_ordinary_rag']['avg_total_tokens']:>16.1f}  | {summary_metrics['arm_3_contrastive_memory']['avg_total_tokens']:>16.1f} ")
    print(f"{'Avg Task Latency (ms)':<28} | {summary_metrics['arm_0_no_memory']['avg_latency_ms']:>16.1f}  | {summary_metrics['arm_1_ordinary_rag']['avg_latency_ms']:>16.1f}  | {summary_metrics['arm_3_contrastive_memory']['avg_latency_ms']:>16.1f} ")
    print("=" * 90)
    print(f"Permanent summary saved to: {summary_file}")

    return summary_metrics


if __name__ == "__main__":
    run_preliminary_comparison()
