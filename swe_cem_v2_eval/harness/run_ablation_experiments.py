#!/usr/bin/env python3
"""
Unified Ablation Experiment Harness for SWE Defect Trajectories (E1 to E6).
Evaluates:
  - E1: Current contrastive memory (Baseline control)
  - E2: Improved episode quality (Clean declarative code deltas, no tool syntax)
  - E3: Applicability-aware retrieval (6-category error taxonomy + novelty dampening)
  - E4: Phase-aware retrieval (5 lifecycle phases with token budgeting)
  - E5: Selective interception (Hard constraints blocked at runtime, decoupled from Pass@1)
  - E6: Full combined system (E2 + E3 + E4 + E5)

Enforces:
  - Decoupled trap interception (interceptions update TAR only, never Pass@1)
  - Ground-truth test execution logging (capturing pytest stdout/stderr for every task)
  - Pinned parameters (GLM 5.3 Flash, temperature=0.0, seed=42, top_p=1.0)
  - Real-time safety budget circuit-breaker ($1.00 USD)
"""

import json
import math
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

HARNESS_DIR = Path(__file__).resolve().parent
EVAL_ROOT = HARNESS_DIR.parent
BASELINE_DIR = EVAL_ROOT.parent / "pi_baseline"
SWE_RIGOROUS_DIR = EVAL_ROOT.parent / "swe_rigorous_eval"

sys.path.insert(0, str(HARNESS_DIR))
sys.path.insert(0, str(BASELINE_DIR / "harness"))
sys.path.insert(0, str(BASELINE_DIR / "harness" / "memory"))
sys.path.insert(0, str(BASELINE_DIR / "harness" / "hooks"))

from glm_provider_client import GLM53FlashClient
from benchmark_loader import ReadOnlyMemoryProxy
from test_execution_runner import TestExecutionRunner
from cem_v2_store import CEMv2MemoryStore
from cem_v2_injector import CEMv2ContextInjector
from cem_v2_interceptor import CEMv2TrapInterceptor
from harvest_cem_v2_memory import CEMv2Harvester

# Baseline store from research/swe_rigorous_eval
from contrastive_store import ContrastiveMemoryStore
from context_injector import ContextInjector
from trap_interceptor import TrapInterceptor


class BudgetExceededError(RuntimeError):
    pass


class AblationSuiteEvaluator:
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = str(EVAL_ROOT / "config" / "eval_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.max_budget_usd = self.config.get("max_budget_usd", 1.00)
        self.cumulative_cost_usd = 0.0
        self.cost_lock = threading.Lock()
        self.abort_requested = False

        self.client = GLM53FlashClient()
        self.test_runner = TestExecutionRunner(test_logs_root=EVAL_ROOT / "test_logs")
        self.runs_root = EVAL_ROOT / "runs"
        self.logs_root = EVAL_ROOT / "logs"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.logs_root.mkdir(parents=True, exist_ok=True)

        # Baseline store (v1)
        self.base_db_path = str(SWE_RIGOROUS_DIR / "memory" / "swe_contrastive_memory.sqlite")
        self.base_store = ReadOnlyMemoryProxy(ContrastiveMemoryStore(db_path=self.base_db_path))
        self.base_injector = ContextInjector(store=self.base_store, max_context_tokens=1000, top_k=3)

        # Improved store (v2)
        self.v2_db_path = str(EVAL_ROOT / "memory" / "cem_v2_contrastive_memory.sqlite")
        self.v2_store = ReadOnlyMemoryProxy(CEMv2MemoryStore(db_path=self.v2_db_path))
        self.v2_injector = CEMv2ContextInjector(store=self.v2_store, max_context_tokens=800, top_k=3, score_threshold=0.68)

        # Eval tasks
        self.eval_file = EVAL_ROOT / "benchmarks" / "eval_swe_stream.json"
        with open(self.eval_file, "r", encoding="utf-8") as f:
            self.eval_tasks = json.load(f)

    def _check_budget(self, cost: float, iid: str, exp_id: str):
        with self.cost_lock:
            self.cumulative_cost_usd += cost
            if self.cumulative_cost_usd > self.max_budget_usd:
                self.abort_requested = True
                raise BudgetExceededError(f"API Spend (${self.cumulative_cost_usd:.4f}) exceeded ceiling (${self.max_budget_usd:.4f}) on {exp_id}/{iid}")

    @staticmethod
    def _extract_full_response_text(resp_dict: Dict[str, Any]) -> str:
        """Extracts complete text from both content and reasoning_content."""
        content = resp_dict.get("content") or ""
        reasoning = resp_dict.get("reasoning_content") or ""
        if reasoning and content:
            return f"{reasoning}\n\n{content}".strip()
        return reasoning if reasoning else content

    def evaluate_task(self, task: Dict[str, Any], exp_id: str, seed: int = 42) -> Dict[str, Any]:
        """Evaluates a single task under a specific ablation experiment configuration."""
        iid = task["instance_id"]
        repo = task["repo"]
        run_dir = self.runs_root / exp_id / f"{iid}_seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_dir / "summary.json"

        # Checkpoint resume
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                return json.load(f)

        instruction = task.get("instruction", "")
        trap_indicator = task.get("trap_indicator", "")
        error_category = CEMv2Harvester.classify_error_category(f"{instruction} {trap_indicator}")

        # Configure prompts and context based on experiment ID
        base_messages = [
            {
                "role": "system",
                "content": (
                    f"You are an expert software engineer resolving a GitHub defect in {repo}.\n"
                    "Analyze the defect, identify the root cause, and provide the exact code modification or patch."
                )
            },
            {
                "role": "user",
                "content": f"Repository: {repo}\nIssue ID: {iid}\n\nIssue Description:\n{instruction}"
            }
        ]

        query_text = f"{repo} {instruction[:300]}"
        scope = {"language": "python", "repository": repo}
        context_tokens = 0
        retrieved_episodes = []
        trap_intercepted = False
        trap_triggered = False

        # --- ABLATION CONFIGURATIONS ---
        if exp_id == "E1_baseline":
            # E1: Current baseline memory (monolithic, raw tool syntax)
            base_messages[0]["content"] += "\nFollow the contrastive episodic guidance strictly."
            messages, context_tokens, retrieved_episodes = self.base_injector.inject(
                messages=base_messages,
                query_text=query_text,
                diagnostic_error=trap_indicator,
                scope=scope
            )

        elif exp_id == "E2_episode_quality":
            # E2: Clean declarative code deltas, no tool call syntax, output format guard
            base_messages[0]["content"] += "\nImportant: Provide direct, fully-formed code implementations. Do NOT output interactive tool call XML tags (e.g., <tool_call>)."
            # Retrieve from v2 store without taxonomy filter, without phase awareness
            messages, context_tokens, retrieved_episodes = self.v2_injector.inject_phase_context(
                messages=base_messages,
                query_text=query_text,
                diagnostic_error=trap_indicator,
                execution_phase="any",
                error_category="GeneralDefect",
                tool_affinity="any",
                scope=scope,
                apply_novelty_dampening=False
            )

        elif exp_id == "E3_applicability":
            # E3: E2 + Error taxonomy matching + Novelty dampening
            base_messages[0]["content"] += "\nImportant: Provide direct, fully-formed code implementations. Do NOT output interactive tool call XML tags (e.g., <tool_call>)."
            messages, context_tokens, retrieved_episodes = self.v2_injector.inject_phase_context(
                messages=base_messages,
                query_text=query_text,
                diagnostic_error=trap_indicator,
                execution_phase="any",
                error_category=error_category,
                tool_affinity="edit",
                scope=scope,
                apply_novelty_dampening=True
            )

        elif exp_id == "E4_phase_aware":
            # E4: E2 + 5 Lifecycle phases with phase token budgeting
            base_messages[0]["content"] += "\nImportant: Provide direct, fully-formed code implementations. Do NOT output interactive tool call XML tags (e.g., <tool_call>)."
            messages, context_tokens, retrieved_episodes = self.v2_injector.inject_phase_context(
                messages=base_messages,
                query_text=query_text,
                diagnostic_error=trap_indicator,
                execution_phase="code_editing",
                error_category="GeneralDefect",
                tool_affinity="any",
                scope=scope,
                apply_novelty_dampening=False
            )

        elif exp_id == "E5_selective_interception":
            # E5: E2 + Dual-tier selective interception (runtime hard constraint blocking)
            base_messages[0]["content"] += "\nImportant: Provide direct, fully-formed code implementations. Do NOT output interactive tool call XML tags (e.g., <tool_call>)."
            messages, context_tokens, retrieved_episodes = self.v2_injector.inject_phase_context(
                messages=base_messages,
                query_text=query_text,
                diagnostic_error=trap_indicator,
                execution_phase="any",
                error_category="GeneralDefect",
                tool_affinity="any",
                scope=scope,
                apply_novelty_dampening=False
            )

        elif exp_id == "E6_combined_v2":
            # E6: Full system: E2 + E3 + E4 + E5
            base_messages[0]["content"] += "\nImportant: Provide direct, fully-formed code implementations. Do NOT output interactive tool call XML tags (e.g., <tool_call>)."
            messages, context_tokens, retrieved_episodes = self.v2_injector.inject_phase_context(
                messages=base_messages,
                query_text=query_text,
                diagnostic_error=trap_indicator,
                execution_phase="code_editing",
                error_category=error_category,
                tool_affinity="edit",
                scope=scope,
                apply_novelty_dampening=True
            )
        else:
            messages = base_messages

        # Model invocation
        t0 = time.perf_counter()
        resp_log = self.client.generate(messages, seed=seed, arm_id=exp_id, task_id=iid)
        latency = (time.perf_counter() - t0) * 1000.0

        full_content = self._extract_full_response_text(resp_log.get("response", {}))

        # Runtime trap evaluation & selective interception (E5 & E6)
        if exp_id in ["E5_selective_interception", "E6_combined_v2"]:
            hard_traps = self.v2_injector.extract_hard_constraints(retrieved_episodes)
            interceptor = CEMv2TrapInterceptor(active_hard_constraints=hard_traps)
            blocked, mitigation, trap_meta = interceptor.evaluate_action_or_content(full_content, task, tool_name="edit")
            if blocked:
                trap_intercepted = True

        # Check fatal trap triggers for TAR calculation
        trap_triggered = ("recursion" in trap_indicator.lower() and "recursion" in full_content.lower() and "base case" not in full_content.lower()) or \
                         ("deadlock" in trap_indicator.lower() and "time.sleep" in full_content.lower()) or \
                         ("unshielded action" in trap_indicator.lower() and "wrong_target" in full_content.lower())

        # Ground-truth test execution oracle (captures raw pytest logs)
        test_result = self.test_runner.execute_test(
            experiment_id=exp_id,
            task=task,
            model_completion=full_content
        )
        passed = test_result["passed"]  # Strictly exit_code == 0

        self._check_budget(resp_log.get("step_cost_usd", 0.0), iid, exp_id)

        summary = {
            "instance_id": iid,
            "repo": repo,
            "experiment_id": exp_id,
            "seed": seed,
            "complexity": task.get("complexity", "medium"),
            "error_category": error_category,
            "passed": passed,  # Strictly decoupled: test passed
            "trap_avoided": not trap_triggered or trap_intercepted,
            "trap_intercepted": trap_intercepted,
            "has_tool_call_syntax": "<tool_call>" in full_content,
            "retrieved_episodes_count": len(retrieved_episodes),
            "retrieved_episode_ids": [ep.get("episodeId") for ep in retrieved_episodes],
            "prompt_tokens": resp_log["prompt_tokens"] + context_tokens,
            "completion_tokens": resp_log["completion_tokens"],
            "memory_injected_tokens": context_tokens,
            "total_tokens": resp_log["total_tokens"] + context_tokens,
            "latency_ms": latency,
            "cost_usd": resp_log["step_cost_usd"],
            "test_exit_code": test_result["exit_code"],
            "test_log_path": test_result["log_path"],
            "run_dir": str(run_dir)
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    def run_experiment(self, exp_id: str, sample_size: Optional[int] = None, workers: int = 6) -> List[Dict[str, Any]]:
        tasks = self.eval_tasks[:sample_size] if sample_size else self.eval_tasks
        n = len(tasks)
        print(f"\n>>> Running Experiment [{exp_id}] (N={n}, Workers={workers}) <<<", flush=True)
        results = [None] * n

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(self.evaluate_task, t, exp_id): idx
                for idx, t in enumerate(tasks)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                t = tasks[idx]
                iid = t["instance_id"]
                try:
                    res = future.result()
                    results[idx] = res
                    status = "PASS" if res["passed"] else "FAIL"
                    tar = "YES" if res["trap_avoided"] else "NO"
                    print(f"  [{idx + 1}/{n}] {iid}: [{status}] (TAR: {tar} | ExitCode: {res['test_exit_code']} | Spend: ${self.cumulative_cost_usd:.4f})", flush=True)
                except BudgetExceededError as be:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise be
                except Exception as e:
                    print(f"  [{idx + 1}/{n}] {iid} ERROR: {e}", flush=True)
                    results[idx] = {
                        "instance_id": iid,
                        "experiment_id": exp_id,
                        "passed": False,
                        "trap_avoided": False,
                        "cost_usd": 0.0,
                        "error": str(e)
                    }

        return [r for r in results if r is not None]


if __name__ == "__main__":
    suite = AblationSuiteEvaluator()
    # Test single task calibration
    res = suite.evaluate_task(suite.eval_tasks[0], exp_id="E1_baseline")
    print(f"Calibration task evaluated: {res['instance_id']} | Passed: {res['passed']} | Log: {res['test_log_path']}")
