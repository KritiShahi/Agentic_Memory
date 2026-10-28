#!/usr/bin/env python3
"""
Mid-Flight beforeToolCall Trap Interceptor (Architecture 1).

Subsystem Component for Phase 2:
- Evaluates candidate tool calls (bash commands, file edits) before OS execution.
- Matches tool arguments against known negative action traps (tau-) from retrieved episodes.
- If a match is detected:
    1. Blocks execution before OS dispatch (prevents environment corruption).
    2. Injects a real-time steering prompt directing the agent to the proven pivot fix.
    3. Records the trap interception in telemetry for Trap Avoidance Rate (TAR) calculation.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


class TrapInterceptor:
    """
    Evaluates and intercepts tool invocations matching known failure traps.
    """
    def __init__(self, active_traps: Optional[List[Dict[str, Any]]] = None):
        self.active_traps: List[Dict[str, Any]] = active_traps if active_traps is not None else []
        self.intercepted_history: List[Dict[str, Any]] = []
        self.total_evaluations: int = 0

    def register_traps(self, traps: List[Dict[str, Any]]) -> None:
        """Registers a list of active trap patterns for the session."""
        self.active_traps.extend(traps)

    def clear_traps(self) -> None:
        """Clears registered traps."""
        self.active_traps = []
        self.intercepted_history = []
        self.total_evaluations = 0

    @staticmethod
    def _normalize_str(text: str) -> str:
        if not text:
            return ""
        # Lowercase, strip extra whitespace, normalize quotes
        cleaned = text.lower().replace('"', "'")
        return re.sub(r"\s+", " ", cleaned).strip()

    def _matches_action(
        self,
        cand_tool: str,
        cand_args: Dict[str, Any],
        trap_tool: str,
        trap_args: Dict[str, Any]
    ) -> bool:
        """
        Determines if candidate tool call matches a known negative action trap.
        """
        if cand_tool.lower() != trap_tool.lower():
            return False

        # 1. Exact argument match
        if cand_args == trap_args:
            return True

        # 2. Bash command matching
        if cand_tool.lower() in ["bash", "cmd", "exec"]:
            cand_cmd = str(cand_args.get("command", cand_args.get("cmd", "")))
            trap_cmd = str(trap_args.get("command", trap_args.get("cmd", "")))
            
            norm_cand = self._normalize_str(cand_cmd)
            norm_trap = self._normalize_str(trap_cmd)
            
            if norm_cand and norm_trap and (norm_cand == norm_trap or norm_trap in norm_cand):
                return True

        # 3. File edit matching (e.g. inserting broken configuration)
        if cand_tool.lower() in ["edit", "write", "file_edit", "str_replace"]:
            cand_path = str(cand_args.get("path", cand_args.get("file", "")))
            trap_path = str(trap_args.get("path", trap_args.get("file", "")))
            
            # If targeting the same file
            if cand_path and trap_path and (Path(cand_path).name == Path(trap_path).name):
                cand_content = self._normalize_str(str(cand_args.get("content", cand_args.get("new_str", ""))))
                trap_content = self._normalize_str(str(trap_args.get("content", trap_args.get("new_str", ""))))
                
                if trap_content and trap_content in cand_content:
                    return True

        return False

    def evaluate_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Evaluates a tool call before OS dispatch.
        Returns: (should_block: bool, mitigation_prompt: Optional[str], matching_trap: Optional[Dict])
        """
        self.total_evaluations += 1

        for trap in self.active_traps:
            trap_tool = trap.get("tool", "")
            trap_args = trap.get("args", {})

            if self._matches_action(tool_name, tool_args, trap_tool, trap_args):
                ep_id = trap.get("episodeId", "ep_unknown")
                warning = trap.get("invariantWarning", "Executing this command leads to a known failure.")
                fix = trap.get("provenFix", "Apply verified pivot fix instead.")
                pivot = trap.get("causalPivot", "")

                mitigation_message = (
                    f"EXECUTION BLOCKED [Episodic Trap Intercept]:\n"
                    f"Action `{tool_name}({json.dumps(tool_args)})` was aborted before OS execution.\n"
                    f"In previous episode [{ep_id}], this exact action caused a verified failure.\n"
                    f"• Invariant Warning: {warning}\n"
                    f"• Proven Pivot Fix:   {fix}\n"
                )
                if pivot:
                    mitigation_message += f"• Causal Pivot Delta (Δpivot):\n  {pivot}\n"
                mitigation_message += "You MUST apply the proven pivot strategy instead of repeating this known trap."

                intercept_record = {
                    "tool": tool_name,
                    "args": tool_args,
                    "matchingTrap": trap,
                    "mitigationMessage": mitigation_message
                }
                self.intercepted_history.append(intercept_record)
                return True, mitigation_message, trap

        return False, None, None

    @property
    def trap_avoidance_metrics(self) -> Dict[str, Any]:
        """Returns statistics on trap evaluations and interceptions."""
        return {
            "total_evaluations": self.total_evaluations,
            "traps_intercepted": len(self.intercepted_history),
            "interception_rate": (len(self.intercepted_history) / max(1, self.total_evaluations))
        }


if __name__ == "__main__":
    interceptor = TrapInterceptor()
    print("TrapInterceptor initialized successfully.")
