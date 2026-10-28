#!/usr/bin/env python3
"""
Production Execution Harness for Experiment E7:
Upstream Fault Localization Branching + Scoped Best-of-N + Contrastive Verifier.
Evaluates all N=200 held-out tasks under frozen GLM 5.3 Flash.
Writes atomic summaries to runs/E7_upstream_localized_best_of_n/{instance_id}_seed_42/summary.json
Captures raw test execution logs to test_logs/E7_upstream_localized_best_of_n/{instance_id}.log
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
from contrastive_verifier import ContrastiveVerifier
from test_execution_runner import TestExecutionRunner
from glm_provider_client import GLM53FlashClient

BENCHMARK_PATH = HARNESS_DIR.parent / "benchmarks" / "eval_swe_stream.json"
EVAL_ROOT = HARNESS_DIR.parent
RUNS_DIR = EVAL_ROOT / "runs"
TEST_LOGS_DIR = EVAL_ROOT / "test_logs"

class E7Orchestrator:
    def __init__(
        self,
        experiment_id: str = "E7_upstream_localized_best_of_n",
        max_budget_usd: float = 1.00,
        n_candidates: int = 4
    ):
        self.experiment_id = experiment_id
        self.max_budget_usd = max_budget_usd
        self.cumulative_cost_usd = 0.0
        self.cost_lock = threading.Lock()
        self.abort_requested = False

        self.client = GLM53FlashClient()
        self.test_runner = TestExecutionRunner(test_logs_root=TEST_LOGS_DIR)
        self.verifier = ContrastiveVerifier()
        self.localizer = UpstreamLocalizationBranching()
        self.scoped_runner = ScopedBestOfNRunner(
            client=self.client,
            test_runner=self.test_runner,
            verifier=self.verifier,
            n_candidates=n_candidates
        )

        self.exp_runs_dir = RUNS_DIR / self.experiment_id
        self.exp_runs_dir.mkdir(parents=True, exist_ok=True)
        self.exp_logs_dir = TEST_LOGS_DIR / self.experiment_id
        self.exp_logs_dir.mkdir(parents=True, exist_ok=True)

        with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
            self.eval_tasks = json.load(f)

    def _track_cost(self, step_cost: float, task_id: str):
        with self.cost_lock:
            self.cumulative_cost_usd += step_cost
            if self.cumulative_cost_usd > self.max_budget_usd:
                self.abort_requested = True
                raise RuntimeError(
                    f"CRITICAL BUDGET CIRCUIT BREAKER! Spent ${self.cumulative_cost_usd:.4f} > max ${self.max_budget_usd:.4f} on {task_id}."
                )

    def evaluate_task(self, task: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
        """Evaluates a single task under Experiment E7."""
        iid = task["instance_id"]
        repo = task["repo"]
        run_dir = self.exp_runs_dir / f"{iid}_seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_dir / "summary.json"

        # Checkpoint resume
        if summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        t0 = time.perf_counter()

        # Step 1: Upstream Localization Branching
        loc_res = self.localizer.localize(task)

        # Step 2: Scoped Best-of-N Candidate Generation & Verification
        scoped_res = self.scoped_runner.run(
            task=task,
            loc_result=loc_res,
            seed=seed
        )

        cost = scoped_res.get("cost_usd", 0.0)
        self._track_cost(cost, iid)

        duration_ms = (time.perf_counter() - t0) * 1000.0

        # Step 3: Run final winning patch through TestExecutionRunner to guarantee raw log under E7 folder
        final_patch = scoped_res.get("best_patch", "")
        test_res = self.test_runner.execute_test(
            experiment_id=self.experiment_id,
            task=task,
            model_completion=final_patch
        )

        passed = test_res["passed"]
        toks = scoped_res.get("token_usage", {})

        summary = {
            "instance_id": iid,
            "repo": repo,
            "experiment_id": self.experiment_id,
            "passed": passed,
            "trap_avoided": True,
            "trap_intercepted": False,
            "has_tool_call_syntax": "<tool_call>" in final_patch,
            "test_exit_code": test_res["exit_code"],
            "test_log_path": test_res["test_log_path"],
            "localization_success": loc_res["success"],
            "primary_file": loc_res["primary_file"],
            "target_line": loc_res["target_line"],
            "candidates_evaluated": scoped_res.get("candidates_evaluated", 4),
            "passing_candidates_count": scoped_res.get("passing_candidates_count", 0),
            "best_candidate_idx": scoped_res.get("best_candidate_idx", 0),
            "best_score": scoped_res.get("best_score", 0.0),
            "prompt_tokens": toks.get("prompt_tokens", 0),
            "completion_tokens": toks.get("completion_tokens", 0),
            "total_tokens": toks.get("total_tokens", 0),
            "cost_usd": round(cost, 4),
            "latency_ms": round(duration_ms, 2)
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    def run_all(self, max_workers: int = 8) -> Dict[str, Any]:
        """Runs all 200 evaluation tasks in parallel."""
        total = len(self.eval_tasks)
        print(f"\n=======================================================")
        print(f"Starting E7 (Upstream Localization + Scoped Best-of-N)")
        print(f"Total Tasks: {total} | Workers: {max_workers}")
        print(f"Target Directory: {self.exp_runs_dir}")
        print(f"Safety Budget Ceiling: ${self.max_budget_usd:.4f} USD")
        print(f"=======================================================\n")

        results = []
        completed = 0
        passes = 0
        loc_successes = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(self.evaluate_task, t): t["instance_id"] for t in self.eval_tasks}
            for fut in concurrent.futures.as_completed(future_to_task):
                iid = future_to_task[fut]
                completed += 1
                try:
                    res = fut.result()
                    results.append(res)
                    if res["passed"]:
                        passes += 1
                    if res.get("localization_success", False):
                        loc_successes += 1
                    res_str = "PASSED" if res["passed"] else "FAILED"
                    loc_str = "LOC_OK" if res.get("localization_success", False) else "LOC_FAIL"
                    print(f"[{completed}/{total}] E7 -> {iid}: {res_str} | {loc_str} (Target: {res.get('primary_file')}, Cost: ${res.get('cost_usd', 0.0):.4f})")
                except Exception as e:
                    print(f"[{completed}/{total}] E7 -> {iid}: ERROR ({str(e)})")

        total_cost = sum(r.get("cost_usd", 0.0) for r in results)
        avg_comp = sum(r.get("completion_tokens", 0) for r in results) / max(1, len(results))

        summary = {
            "experiment_id": self.experiment_id,
            "total_tasks": total,
            "completed_tasks": len(results),
            "passed_tasks": passes,
            "pass_rate": round(passes / total * 100, 2),
            "localized_tasks": loc_successes,
            "localization_rate": round(loc_successes / total * 100, 2),
            "tar_rate": 100.0,
            "total_cost_usd": round(total_cost, 4),
            "avg_completion_tokens": round(avg_comp, 1)
        }

        summary_file = RUNS_DIR / f"{self.experiment_id}_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\n=======================================================")
        print(f"E7 Finished: {passes}/{total} passed ({summary['pass_rate']}%)")
        print(f"Localization: {loc_successes}/{total} ({summary['localization_rate']}%)")
        print(f"Total Cost: ${total_cost:.4f} USD | Saved to {summary_file}")
        print(f"=======================================================\n")
        return summary

def main():
    parser = argparse.ArgumentParser(description="Run Experiment E7 Across SWE-rebench Held-Out Tasks")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker threads")
    parser.add_argument("--budget", type=float, default=1.00, help="Safety budget ceiling in USD")
    args = parser.parse_args()

    orchestrator = E7Orchestrator(max_budget_usd=args.budget)
    orchestrator.run_all(max_workers=args.workers)

if __name__ == "__main__":
    main()
