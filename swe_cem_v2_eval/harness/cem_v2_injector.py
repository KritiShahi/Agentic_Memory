#!/usr/bin/env python3
"""
CEM-v2 Multi-Phase Context Injection Engine.

Features:
- Supports 5 lifecycle phases: [planning, code_editing, test_execution, recovery, verification]
- Non-Coercive Advisory Framing: Preserves agent reasoning while injecting structural insights.
- Format Guard: Explicitly commands outputting clean patch code, preventing <tool_call> XML syntax dropouts.
- Strict Token Budgeting: Enforces max_context_tokens (default <= 800 tokens, top-k <= 3).
"""

import copy
import json
import math
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

HARNESS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS_DIR))

from cem_v2_store import CEMv2MemoryStore


class CEMv2ContextInjector:
    def __init__(
        self,
        store: Optional[CEMv2MemoryStore] = None,
        max_context_tokens: int = 800,
        top_k: int = 3,
        score_threshold: float = 0.68
    ):
        self.store = store if store is not None else CEMv2MemoryStore()
        self.max_context_tokens = max_context_tokens
        self.top_k = top_k
        self.score_threshold = score_threshold

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return max(1, math.ceil(len(text) / 4))

    def format_phase_advisory_block(
        self,
        retrieved_episodes: List[Dict[str, Any]],
        phase: str = "code_editing"
    ) -> str:
        """Formats retrieved episodes into structured advisory guidance."""
        if not retrieved_episodes:
            return ""

        phase_headers = {
            "planning": "ARCHITECTURAL PLANNING & DEFECT ISOLATION",
            "code_editing": "CODE EDITING & BOUNDARY INVARIANT PRESERVATION",
            "test_execution": "TEST ASSERTION & REPRODUCTION GUIDANCE",
            "recovery": "COUNTERFACTUAL RECOVERY & TRAP MITIGATION",
            "verification": "PRE-SUBMISSION VERIFICATION CRITERIA"
        }
        header_title = phase_headers.get(phase, phase.upper())

        blocks = [
            f"=== [CEM-v2 ADVISORY GUIDANCE: {header_title}] ===",
            "The following patterns and code deltas are derived from verified defect resolutions in analogous codebases.",
            "Use them as advisory heuristics to avoid regressions while diagnosing and implementing your fix.",
            "OUTPUT REQUIREMENT: Provide the exact code modification or patch. Do NOT emit interactive agent tool call XML tags (e.g., <tool_call>).\n"
        ]

        for idx, ep in enumerate(retrieved_episodes, 1):
            lesson = ep.get("lesson", {})
            err_cat = ep.get("errorCategory", "GeneralDefect")
            score = ep.get("score", 1.0)
            code_delta = ep.get("codeDeltaSummary", "")

            blocks.append(f"--- Insight #{idx} (Archetype: {err_cat} | Relevance: {score:.2f}) ---")
            if ep.get("diagnosticError"):
                blocks.append(f"• Problem Context: {ep['diagnosticError']}")

            blocks.append(f"• Invariant Trap to Avoid:\n  {lesson.get('invariantConstraint', 'Avoid unshielded naive modifications.')}")
            blocks.append(f"• Actionable Resolution Pattern:\n  {lesson.get('actionablePrescription', 'Apply verified boundary check.')}")

            if code_delta and len(code_delta.strip()) > 5:
                blocks.append(f"• Reference Code Delta (Δpivot):\n{code_delta}")

            blocks.append("")

        blocks.append("================================================================================")
        return "\n".join(blocks)

    def inject_phase_context(
        self,
        messages: List[Dict[str, Any]],
        query_text: str,
        diagnostic_error: str = "",
        execution_phase: str = "code_editing",
        error_category: Optional[str] = None,
        tool_affinity: str = "edit",
        scope: Optional[Dict[str, str]] = None,
        apply_novelty_dampening: bool = True
    ) -> Tuple[List[Dict[str, Any]], int, List[Dict[str, Any]]]:
        """
        Retrieves phase-specific episodes and injects non-coercive advisory guidance.
        Returns: (augmented_messages, injected_tokens, retrieved_episodes)
        """
        retrieved = self.store.retrieve_with_applicability(
            query_text=query_text,
            diagnostic_error=diagnostic_error,
            execution_phase=execution_phase,
            error_category=error_category,
            tool_affinity=tool_affinity,
            scope=scope,
            top_k=self.top_k,
            max_tokens=self.max_context_tokens,
            score_threshold=self.score_threshold,
            apply_novelty_dampening=apply_novelty_dampening
        )

        if not retrieved:
            return messages, 0, []

        # Enforce hard token ceiling
        while retrieved and self.estimate_tokens(self.format_phase_advisory_block(retrieved, phase=execution_phase)) > self.max_context_tokens:
            retrieved.pop()

        if not retrieved:
            return messages, 0, []

        advisory_block = self.format_phase_advisory_block(retrieved, phase=execution_phase)
        injected_tokens = self.estimate_tokens(advisory_block)

        augmented = copy.deepcopy(messages)
        injected = False
        for msg in augmented:
            if msg.get("role") == "system":
                msg["content"] = f"{msg['content']}\n\n{advisory_block}"
                injected = True
                break

        if not injected and augmented:
            augmented[0]["content"] = f"{advisory_block}\n\n{augmented[0]['content']}"

        return augmented, injected_tokens, retrieved

    @staticmethod
    def extract_hard_constraints(retrieved_episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extracts registered Hard Constraints for runtime interceptor blocking."""
        hard_traps = []
        for ep in retrieved_episodes:
            if ep.get("constraintType") == "HARD_CONSTRAINT":
                lesson = ep.get("lesson", {})
                hard_traps.append({
                    "episodeId": ep.get("episodeId"),
                    "invariantWarning": lesson.get("invariantConstraint", ""),
                    "provenFix": lesson.get("actionablePrescription", ""),
                    "tool": ep.get("toolAffinity", "edit"),
                    "codeDelta": ep.get("codeDeltaSummary", "")
                })
        return hard_traps
