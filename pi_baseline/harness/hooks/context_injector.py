#!/usr/bin/env python3
"""
Pre-Flight Context Injection Hook (Architecture 1).

Subsystem Component for Phase 2:
- Triggered at session initialization / turn 0 before prompt compilation.
- Queries SQLite memory store using task description, diagnostic signatures, and scope.
- Enforces strict hard token budget (B_ctx <= 1000 tokens, top-k <= 3).
- Formats contrastive episodes into actionable invariant constraints and pivot rules.
- Injects structured guidance into the agent's prompt context.
"""

import copy
import json
import math
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Path setup
HOOKS_DIR = Path(__file__).resolve().parent
HARNESS_DIR = HOOKS_DIR.parent
MEMORY_DIR = HARNESS_DIR / "memory"
sys.path.insert(0, str(MEMORY_DIR))

from contrastive_store import ContrastiveMemoryStore


class ContextInjector:
    """
    Manages pre-flight retrieval and injection of contrastive episodic memories.
    """
    def __init__(
        self,
        store: Optional[ContrastiveMemoryStore] = None,
        max_context_tokens: int = 1000,
        top_k: int = 3,
        score_threshold: float = 0.72
    ):
        self.store = store if store is not None else ContrastiveMemoryStore()
        self.max_context_tokens = max_context_tokens
        self.top_k = top_k
        self.score_threshold = score_threshold

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return max(1, math.ceil(len(text) / 4))

    def format_contrastive_block(self, retrieved_episodes: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved contrastive episodes into structured prompt text.
        """
        if not retrieved_episodes:
            return ""

        blocks = [
            "=== [EPISODIC MEMORY: CONTRASTIVE GUIDANCE] ===",
            "The following verified experiences from previous problem-solving sessions are relevant to this task.",
            "You MUST strictly follow the Invariant Warnings and apply the Proven Pivot Fixes.\n"
        ]

        for idx, ep in enumerate(retrieved_episodes, 1):
            lesson = ep.get("lesson", {})
            div_point = ep.get("divergencePoint", {})
            verif = ep.get("verification", {})
            score = ep.get("hybridScore", 1.0)
            prob_sig = ep.get("problemSignature", "sig")

            blocks.append(f"--- Experience #{idx} (Signature: {prob_sig} | Relevance Score: {score:.2f}) ---")
            if ep.get("diagnosticError"):
                blocks.append(f"• Problem Diagnostic: {ep['diagnosticError']}")
            
            blocks.append(f"• INVARIANT WARNING (Anti-Pattern Trap):\n  {lesson.get('invariantConstraint', 'Avoid known failed approach.')}")
            blocks.append(f"• PROVEN PIVOT FIX (What To Do Instead):\n  {lesson.get('actionablePrescription', 'Apply verified pivot.')}")
            
            if div_point.get("causalPivotDelta"):
                blocks.append(f"• Causal Pivot Delta (Δpivot):\n  {div_point['causalPivotDelta']}")
                
            blocks.append(f"• Verification Evidence: {verif.get('testsPassed', 0)} passed, {verif.get('testsFailed', 0)} failed.")
            blocks.append("")

        blocks.append("==================================================")
        return "\n".join(blocks)

    def inject(
        self,
        messages: List[Dict[str, Any]],
        query_text: str,
        diagnostic_error: str = "",
        scope: Optional[Dict[str, str]] = None
    ) -> Tuple[List[Dict[str, Any]], int, List[Dict[str, Any]]]:
        """
        Retrieves relevant contrastive episodes and injects them into the message list.
        Returns: (augmented_messages, memory_injected_tokens, retrieved_episodes)
        """
        # 1. Retrieve candidates under strict budget
        retrieved = self.store.retrieve(
            query_text=query_text,
            diagnostic_error=diagnostic_error,
            scope=scope,
            top_k=self.top_k,
            max_tokens=self.max_context_tokens,
            score_threshold=self.score_threshold
        )

        if not retrieved:
            return messages, 0, []

        # 2. Format contrastive block & strictly enforce hard token ceiling
        while retrieved and self.estimate_tokens(self.format_contrastive_block(retrieved)) > self.max_context_tokens:
            retrieved.pop()

        if not retrieved:
            return messages, 0, []

        context_block = self.format_contrastive_block(retrieved)
        injected_tokens = self.estimate_tokens(context_block)

        # 3. Inject into prompt messages
        augmented = copy.deepcopy(messages)
        
        # Inject into system prompt if present, or prepend to first user message
        injected = False
        for msg in augmented:
            if msg.get("role") == "system":
                msg["content"] = f"{msg['content']}\n\n{context_block}"
                injected = True
                break

        if not injected and augmented:
            # Prepend to the first message
            augmented[0]["content"] = f"{context_block}\n\n{augmented[0]['content']}"

        return augmented, injected_tokens, retrieved

    @staticmethod
    def extract_active_traps(retrieved_episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extracts known bad action patterns from retrieved episodes for mid-flight interception.
        """
        active_traps = []
        for ep in retrieved_episodes:
            failed_traj = ep.get("failedTrajectory", {})
            lesson = ep.get("lesson", {})
            div_point = ep.get("divergencePoint", {})
            
            for act in failed_traj.get("actions", []):
                active_traps.append({
                    "episodeId": ep.get("episodeId"),
                    "problemSignature": ep.get("problemSignature"),
                    "tool": act.get("tool", ""),
                    "args": act.get("args", {}),
                    "invariantWarning": lesson.get("invariantConstraint", ""),
                    "provenFix": lesson.get("actionablePrescription", ""),
                    "causalPivot": div_point.get("causalPivotDelta", "")
                })
        return active_traps


if __name__ == "__main__":
    injector = ContextInjector(ContrastiveMemoryStore(":memory:"))
    print("ContextInjector initialized successfully.")
