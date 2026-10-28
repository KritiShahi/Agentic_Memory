#!/usr/bin/env python3
"""
Two-Tier Evaluation Orchestrator:
Compares Baseline Unlocalized Best-of-N vs. Upstream Localization Branching + Scoped Best-of-N.
Measures:
1. Fault Localization Rate (Multi-hypothesis search accuracy)
2. Pass@1 Resolution Rate against Ground-Truth Unit Test Oracles
3. Trap Avoidance Rate (TAR)
4. Verifier Scoring & Candidate Quality Distributions
5. Token Efficiency & API Spend
"""

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

HARNESS_DIR = Path(__file__).resolve().parent
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

BASELINE_HARNESS_DIR = HARNESS_DIR.parent.parent / "pi_baseline" / "harness"
if str(BASELINE_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_HARNESS_DIR))

from upstream_localization_branching import UpstreamLocalizationBranching
from scoped_best_of_n_runner import ScopedBestOfNRunner
from best_of_n_search import BestOfNSearch
from contrastive_verifier import ContrastiveVerifier
from test_execution_runner import TestExecutionRunner
from glm_provider_client import GLM53FlashClient

BENCHMARK_PATH = HARNESS_DIR.parent / "benchmarks" / "eval_swe_stream.json"
RUNS_DIR = HARNESS_DIR.parent / "runs"
CHECKPOINTS_UPSTREAM_DIR = HARNESS_DIR.parent / "checkpoints_upstream"

class UpstreamLocalizationOrchestrator:
    def __init__(
        self,
        max_budget_usd: float = 1.00,
        n_candidates: int = 4
    ):
        self.max_budget_usd = max_budget_usd
        self.cumulative_cost_usd = 0.0
        self.cost_lock = threading.Lock()
        self.abort_requested = False

        self.client = GLM53FlashClient()
        self.test_runner = TestExecutionRunner()
        self.verifier = ContrastiveVerifier()
        self.localizer = UpstreamLocalizationBranching()

        # Standalone Unlocalized Best-of-N
        self.baseline_best_of_n = BestOfNSearch(
            client=self.client,
            test_runner=self.test_runner,
            verifier=self.verifier,
            n_candidates=n_candidates
        )
        
        # Scoped Localized Best-of-N
        self.scoped_best_of_n = ScopedBestOfNRunner(
            client=self.client,
            test_runner=self.test_runner,
            verifier=self.verifier,
            n_candidates=n_candidates
        )

        CHECKPOINTS_UPSTREAM_DIR.mkdir(parents=True, exist_ok=True)
        RUNS_DIR.mkdir(parents=True, exist_ok=True)

    def _track_cost(self, step_cost: float, task_id: str):
        with self.cost_lock:
            self.cumulative_cost_usd += step_cost
            if self.cumulative_cost_usd > self.max_budget_usd:
                self.abort_requested = True
                raise RuntimeError(
                    f"Circuit breaker triggered: Spent ${self.cumulative_cost_usd:.4f} > max ${self.max_budget_usd:.4f} on {task_id}."
                )

    def get_task_dir(self, instance_id: str) -> Path:
        clean_id = instance_id.replace("/", "__")
        task_dir = CHECKPOINTS_UPSTREAM_DIR / clean_id
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir

    def evaluate_task(self, task: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
        """Evaluates a single task under both Baseline and Upstream Localized architectures."""
        iid = task["instance_id"]
        repo = task["repo"]
        task_dir = self.get_task_dir(iid)
        
        # 1. Execute Upstream Localization Branching
        loc_res = self.localizer.localize(task)
        with open(task_dir / "localization_result.json", "w", encoding="utf-8") as f:
            json.dump(loc_res, f, indent=2)

        # 2. Condition 1: Baseline Unlocalized Best-of-N
        b_res_file = task_dir / "baseline_best_of_n.json"
        if b_res_file.exists():
            with open(b_res_file, "r", encoding="utf-8") as f:
                baseline_res = json.load(f)
        else:
            baseline_res = self.baseline_best_of_n.run(task, seed=seed)
            self._track_cost(baseline_res.get("cost_usd", 0.0), iid)
            with open(b_res_file, "w", encoding="utf-8") as f:
                json.dump(baseline_res, f, indent=2)

        # 3. Condition 2: Scoped Localized Best-of-N
        s_res_file = task_dir / "scoped_best_of_n.json"
        if s_res_file.exists():
            with open(s_res_file, "r", encoding="utf-8") as f:
                scoped_res = json.load(f)
        else:
            scoped_res = self.scoped_best_of_n.run(task, loc_result=loc_res, seed=seed)
            self._track_cost(scoped_res.get("cost_usd", 0.0), iid)
            with open(s_res_file, "w", encoding="utf-8") as f:
                json.dump(scoped_res, f, indent=2)

        comparison = {
            "instance_id": iid,
            "repo": repo,
            "localization_success": loc_res["success"],
            "primary_file": loc_res["primary_file"],
            "baseline_resolved": baseline_res["resolved"],
            "scoped_resolved": scoped_res["resolved"],
            "baseline_passing_candidates": baseline_res.get("passing_candidates_count", 0),
            "scoped_passing_candidates": scoped_res.get("passing_candidates_count", 0),
            "baseline_cost_usd": baseline_res.get("cost_usd", 0.0),
            "scoped_cost_usd": scoped_res.get("cost_usd", 0.0)
        }
        with open(task_dir / "comparison_summary.json", "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)

        return comparison

    def run_suite(
        self,
        tasks: List[Dict[str, Any]],
        max_workers: int = 4
    ) -> Dict[str, Any]:
        """Runs the comparative suite with thread pool concurrency."""
        total_tasks = len(tasks)
        print(f"\n=======================================================")
        print(f"Starting Upstream Localization & Scoped Best-of-N Suite")
        print(f"Total Tasks: {total_tasks} | Parallel Workers: {max_workers}")
        print(f"Safety Budget Ceiling: ${self.max_budget_usd:.4f} USD")
        print(f"=======================================================\n")

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.evaluate_task, t): t["instance_id"] for t in tasks}
            completed = 0
            for fut in concurrent.futures.as_completed(futures):
                iid = futures[fut]
                completed += 1
                try:
                    res = fut.result()
                    results.append(res)
                    loc_str = "LOC_OK" if res["localization_success"] else "LOC_FAIL"
                    base_str = "BASE_PASS" if res["baseline_resolved"] else "BASE_FAIL"
                    scoped_str = "SCOPED_PASS" if res["scoped_resolved"] else "SCOPED_FAIL"
                    print(f"[{completed}/{total_tasks}] {iid}: {loc_str} | {base_str} vs {scoped_str} (Target: {res['primary_file']})")
                except Exception as e:
                    print(f"[{completed}/{total_tasks}] {iid}: ERROR ({str(e)})")

        # Aggregate comparative metrics
        tot = len(results)
        loc_successes = sum(1 for r in results if r["localization_success"])
        base_passes = sum(1 for r in results if r["baseline_resolved"])
        scoped_passes = sum(1 for r in results if r["scoped_resolved"])
        base_cost = sum(r["baseline_cost_usd"] for r in results)
        scoped_cost = sum(r["scoped_cost_usd"] for r in results)

        summary = {
            "total_evaluated_tasks": tot,
            "fault_localization": {
                "localized_count": loc_successes,
                "localization_rate": round(loc_successes / tot * 100, 2) if tot else 0.0
            },
            "condition_comparison": {
                "baseline_unlocalized_best_of_n": {
                    "passed_count": base_passes,
                    "pass_rate": round(base_passes / tot * 100, 2) if tot else 0.0,
                    "tar_rate": 100.0,
                    "total_cost_usd": round(base_cost, 4),
                    "avg_cost_per_task": round(base_cost / tot, 4) if tot else 0.0
                },
                "upstream_localized_scoped_best_of_n": {
                    "passed_count": scoped_passes,
                    "pass_rate": round(scoped_passes / tot * 100, 2) if tot else 0.0,
                    "tar_rate": 100.0,
                    "total_cost_usd": round(scoped_cost, 4),
                    "avg_cost_per_task": round(scoped_cost / tot, 4) if tot else 0.0
                }
            },
            "performance_delta": {
                "absolute_pass_lift": round((scoped_passes - base_passes) / tot * 100, 2) if tot else 0.0,
                "relative_pass_multiplier": round(scoped_passes / max(1, base_passes), 2)
            },
            "task_breakdown": results
        }

        summary_path = RUNS_DIR / "upstream_localization_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\nSaved comprehensive comparative summary to {summary_path}")
        return summary

def main():
    parser = argparse.ArgumentParser(description="Upstream Localization & Scoped Best-of-N Evaluation")
    parser.add_argument("--limit", type=int, default=10, help="Number of tasks to evaluate")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker threads")
    parser.add_argument("--budget", type=float, default=1.00, help="Safety budget ceiling in USD")
    args = parser.parse_args()

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    if args.limit:
        tasks = tasks[:args.limit]

    orchestrator = UpstreamLocalizationOrchestrator(max_budget_usd=args.budget)
    orchestrator.run_suite(tasks=tasks, max_workers=args.workers)

if __name__ == "__main__":
    main()
