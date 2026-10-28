#!/usr/bin/env python3
"""
CEM-v2 Dual-Tier Trap Interceptor.
Enforces execution-time Hard Constraints while allowing Soft Recommendations
to remain non-blocking advisory suggestions.
"""

from typing import Dict, Any, List, Optional, Tuple


class CEMv2TrapInterceptor:
    def __init__(self, active_hard_constraints: Optional[List[Dict[str, Any]]] = None):
        self.hard_constraints: List[Dict[str, Any]] = active_hard_constraints or []
        self.intercepted_count = 0

    def register_hard_constraints(self, constraints: List[Dict[str, Any]]):
        self.hard_constraints.extend(constraints)

    def evaluate_action_or_content(
        self,
        content: str,
        task: Dict[str, Any],
        tool_name: str = "edit"
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Evaluates model response content or tool invocation against registered Hard Constraints.
        Returns: (should_block, mitigation_message, matching_trap)
        """
        if not content:
            return False, None, None

        content_lower = content.lower()
        trap_indicator = task.get("trap_indicator", "").lower()

        # Check registered hard constraints
        for trap in self.hard_constraints:
            warning = trap.get("invariantWarning", "").lower()
            if warning and warning in content_lower:
                self.intercepted_count += 1
                mitigation = (
                    f"[CEM-v2 HARD CONSTRAINT VIOLATION BLOCKED]\n"
                    f"Action matches fatal anti-pattern: {trap.get('invariantWarning')}\n"
                    f"MANDATORY COUNTERFACTUAL PIVOT: {trap.get('provenFix')}\n"
                    f"Code delta: {trap.get('codeDelta')}"
                )
                return True, mitigation, trap

        # Check fatal task trap patterns
        if "recursion" in trap_indicator and "recursion" in content_lower and "base case" not in content_lower:
            self.intercepted_count += 1
            mitigation = f"[CEM-v2 HARD CONSTRAINT]: Unbounded recursion blocked. Apply base-case guard."
            return True, mitigation, {"type": "recursion", "provenFix": task.get("counterfactual_pivot")}

        if "deadlock" in trap_indicator and "time.sleep" in content_lower:
            self.intercepted_count += 1
            mitigation = f"[CEM-v2 HARD CONSTRAINT]: Sleep deadlock blocked. Use non-blocking event polling."
            return True, mitigation, {"type": "deadlock", "provenFix": task.get("counterfactual_pivot")}

        if "unshielded action" in trap_indicator and "wrong_target" in content_lower:
            self.intercepted_count += 1
            mitigation = f"[CEM-v2 HARD CONSTRAINT]: Unshielded path modification blocked. Pivot to target module."
            return True, mitigation, {"type": "unshielded_action", "provenFix": task.get("counterfactual_pivot")}

        return False, None, None
