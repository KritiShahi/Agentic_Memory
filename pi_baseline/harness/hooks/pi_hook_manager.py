#!/usr/bin/env python3
"""
Unified Pi Hook Manager for Contrastive Episodic Memory (Architecture 1).

Subsystem Coordinator for Phase 2:
- Connects the Pi agent lifecycle events with memory subsystems:
    1. before_prompt: Pre-flight context retrieval & token-budgeted prompt injection.
    2. before_tool_call: Mid-flight proactive interception of known failed action traps.
    3. on_session_settled: Post-session fork analysis, Delta_pivot extraction, and SQLite consolidation.
    4. on_session_feedback: Reinforcement and Negative Utility Pruning (auto-tombstoning).
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Path setup
HOOKS_DIR = Path(__file__).resolve().parent
HARNESS_DIR = HOOKS_DIR.parent
MEMORY_DIR = HARNESS_DIR / "memory"
sys.path.insert(0, str(MEMORY_DIR))
sys.path.insert(0, str(HOOKS_DIR))

from contrastive_store import ContrastiveMemoryStore
from contrastive_harvester import ContrastiveHarvester
from context_injector import ContextInjector
from trap_interceptor import TrapInterceptor


class PiHookManager:
    """
    Unified manager orchestrating the complete Contrastive Episodic Memory lifecycle.
    """
    def __init__(
        self,
        store: Optional[ContrastiveMemoryStore] = None,
        max_context_tokens: int = 1000,
        top_k: int = 3,
        score_threshold: float = 0.72
    ):
        self.store = store if store is not None else ContrastiveMemoryStore()
        self.harvester = ContrastiveHarvester(self.store)
        self.injector = ContextInjector(
            self.store,
            max_context_tokens=max_context_tokens,
            top_k=top_k,
            score_threshold=score_threshold
        )
        self.interceptor = TrapInterceptor()
        self.active_session_retrieved: List[Dict[str, Any]] = []
        # In-flight step and backtrack history tracking for autonomous harvesting
        self.current_session_steps: List[Dict[str, Any]] = []
        self.in_flight_backtracks: List[Dict[str, Any]] = []

    def before_prompt(
        self,
        messages: List[Dict[str, Any]],
        query_text: str,
        diagnostic_error: str = "",
        scope: Optional[Dict[str, str]] = None
    ) -> Tuple[List[Dict[str, Any]], int, List[Dict[str, Any]]]:
        """
        Pre-flight hook: Retrieves relevant episodes, injects contrastive guidance into prompt,
        and registers active trap patterns for mid-flight monitoring.
        """
        self.interceptor.clear_traps()
        
        augmented_messages, memory_tokens, retrieved_episodes = self.injector.inject(
            messages=messages,
            query_text=query_text,
            diagnostic_error=diagnostic_error,
            scope=scope
        )

        self.active_session_retrieved = retrieved_episodes

        # Register negative action traps for real-time beforeToolCall interception
        if retrieved_episodes:
            traps = self.injector.extract_active_traps(retrieved_episodes)
            self.interceptor.register_traps(traps)

        return augmented_messages, memory_tokens, retrieved_episodes

    def before_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Mid-flight hook: Evaluates a candidate tool action against active negative traps.
        If a trap is matched, blocks execution and returns a mitigation prompt.
        """
        return self.interceptor.evaluate_tool_call(tool_name, tool_args)

    def after_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        exit_code: int = 0,
        output: str = "",
        hypothesis: str = "",
        failure_evidence: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Post-execution hook: Observes tool execution outcome, records the step in
        chronological history, and flags failure events for autonomous backtrack harvesting.
        """
        step_idx = len(self.current_session_steps)
        step_record = {
            "step": step_idx,
            "hypothesis": hypothesis or f"Execute {tool_name}",
            "tool": tool_name,
            "args": tool_args,
            "output": output[:1000] if output else "",
            "exitCode": exit_code
        }

        # Detect and tag failure evidence
        if exit_code != 0 or failure_evidence:
            root_msg = ""
            if output:
                lines = [line.strip() for line in output.strip().splitlines() if line.strip()]
                root_msg = lines[-1] if lines else f"Tool {tool_name} exited with status {exit_code}"
            else:
                root_msg = f"Tool {tool_name} returned exit code {exit_code}"

            step_record["failureEvidence"] = failure_evidence or {
                "errorCategory": f"{tool_name}_execution_error",
                "rootMessage": root_msg
            }
            self.in_flight_backtracks.append(step_record)

        self.current_session_steps.append(step_record)
        return step_record

    def on_session_settled(
        self,
        session_data: Dict[str, Any],
        auto_persist: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Post-session consolidation hook: Extracts contrastive failure-success pairs,
        computes Delta_pivot, synthesizes lessons, and persists into SQLite.
        Automatically supplies in-flight session steps and backtrack history if not explicitly provided.
        """
        # Auto-attach recorded in-flight steps if explicit branches were omitted
        if "failedBranch" not in session_data:
            if "steps" not in session_data and self.current_session_steps:
                session_data["steps"] = list(self.current_session_steps)
            if "backtrackHistory" not in session_data and self.in_flight_backtracks:
                session_data["backtrackHistory"] = list(self.in_flight_backtracks)

        harvested = self.harvester.harvest(session_data, auto_persist=auto_persist)
        
        # Flush session buffer
        self.current_session_steps = []
        self.in_flight_backtracks = []
        return harvested

    def on_session_feedback(
        self,
        episode_id: str,
        success: bool,
        penalty_threshold: int = 2
    ) -> Dict[str, Any]:
        """
        Lifecycle feedback hook: Reinforces successful memory usage or applies
        Negative Utility Pruning when memories cause failures.
        """
        return self.store.record_feedback(
            episode_id=episode_id,
            success=success,
            penalty_threshold=penalty_threshold
        )

    def reset_session(self) -> None:
        """Resets in-flight session state between task executions."""
        self.interceptor.clear_traps()
        self.active_session_retrieved = []
        self.current_session_steps = []
        self.in_flight_backtracks = []

    @property
    def metrics(self) -> Dict[str, Any]:
        """Returns runtime hook metrics."""
        return {
            "trap_interceptor": self.interceptor.trap_avoidance_metrics,
            "active_retrieved_count": len(self.active_session_retrieved)
        }


if __name__ == "__main__":
    manager = PiHookManager(ContrastiveMemoryStore(":memory:"))
    print("PiHookManager unified coordinator initialized successfully.")
