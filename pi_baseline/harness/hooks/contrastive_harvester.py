#!/usr/bin/env python3
"""
Contrastive Episode Harvester & Causal Pivot Extractor.

Subsystem Component for Phase 2:
- Traverses session DAG / step history upon sessionSettled event.
- Identifies lowest common ancestor step between aborted failure branch (tau-)
  and final verified success branch (tau+).
- Computes Causal Pivot Delta (Delta_pivot = action diff + state delta).
- Synthesizes actionable contrastive lesson (Invariant Anti-pattern + Actionable Prescription).
- Atomically writes ContrastiveEpisode into SQLite via ContrastiveMemoryStore.
"""

import json
import os
import re
import sys
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Path setup
HOOKS_DIR = Path(__file__).resolve().parent
HARNESS_DIR = HOOKS_DIR.parent
MEMORY_DIR = HARNESS_DIR / "memory"
sys.path.insert(0, str(MEMORY_DIR))

from contrastive_store import ContrastiveMemoryStore


class ContrastiveHarvester:
    """
    Analyzes session execution trees and extracts contrastive counterfactual episodes.
    """
    def __init__(self, store: Optional[ContrastiveMemoryStore] = None):
        self.store = store if store is not None else ContrastiveMemoryStore()

    @staticmethod
    def find_lowest_common_ancestor(
        failed_branch: List[Dict[str, Any]],
        success_branch: List[Dict[str, Any]]
    ) -> int:
        """
        Finds the fork step index where the agent's strategy diverged.
        """
        min_len = min(len(failed_branch), len(success_branch))
        fork_idx = 0
        for i in range(min_len):
            step_f = failed_branch[i]
            step_s = success_branch[i]
            # If actions or hypotheses differ, divergence occurred at step i
            if (
                step_f.get("hypothesis") != step_s.get("hypothesis") or
                step_f.get("tool") != step_s.get("tool") or
                step_f.get("args") != step_s.get("args")
            ):
                return i
            fork_idx = i + 1
        return fork_idx

    @staticmethod
    def compute_causal_pivot_delta(
        failed_branch: List[Dict[str, Any]],
        success_branch: List[Dict[str, Any]],
        fork_idx: int
    ) -> Dict[str, Any]:
        """
        Computes the action and state differential between the failed and successful paths.
        """
        failed_divergent = failed_branch[fork_idx:] if fork_idx < len(failed_branch) else failed_branch[-1:]
        success_divergent = success_branch[fork_idx:] if fork_idx < len(success_branch) else success_branch[-1:]

        failed_actions = [s.get("tool", "") + "(" + json.dumps(s.get("args", {})) + ")" for s in failed_divergent]
        success_actions = [s.get("tool", "") + "(" + json.dumps(s.get("args", {})) + ")" for s in success_divergent]

        failed_hypo = failed_divergent[0].get("hypothesis", "Initial failed strategy") if failed_divergent else "Failed approach"
        success_hypo = success_divergent[0].get("hypothesis", "Alternative strategy") if success_divergent else "Passing fix"

        # Action diff representation
        action_diff = (
            f"[-] Aborted Actions: {' -> '.join(failed_actions)}\n"
            f"[+] Pivoted Actions: {' -> '.join(success_actions)}"
        )

        return {
            "forkStepIndex": fork_idx,
            "failedDecision": failed_hypo,
            "successfulDecision": success_hypo,
            "causalPivotDelta": action_diff
        }

    @staticmethod
    def synthesize_lesson(
        diagnostic_error: str,
        failed_branch: List[Dict[str, Any]],
        success_branch: List[Dict[str, Any]],
        pivot_delta: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Synthesizes an invariant constraint (anti-pattern) and an actionable prescription.
        """
        failed_step = failed_branch[-1] if failed_branch else {}
        success_step = success_branch[-1] if success_branch else {}

        failed_err = failed_step.get("failureEvidence", {}).get("rootMessage", "Execution failure")
        failed_act_summary = failed_step.get("tool", "") + " " + str(failed_step.get("args", {}))
        success_act_summary = success_step.get("tool", "") + " " + str(success_step.get("args", {}))

        clean_diag = re.sub(r"/\S+/", "", diagnostic_error).strip()[:100]

        invariant_rule = (
            f"When encountering `{clean_diag}`, do NOT attempt: "
            f"`{failed_act_summary}` (Causes: {failed_err})."
        )

        prescription_rule = (
            f"Apply proven pivot: `{success_act_summary}` "
            f"({pivot_delta.get('successfulDecision', 'Verified resolution')})."
        )

        return {
            "invariantConstraint": invariant_rule,
            "actionablePrescription": prescription_rule
        }

    @staticmethod
    def partition_linear_steps(steps: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Autonomously partitions a linear chronological step trace into a counterfactual
        failed branch (tau-) and verified success branch (tau+).
        """
        if not steps or len(steps) < 2:
            return [], []

        # Find the first step that exhibited an error or failure
        first_fail_idx = -1
        for i, s in enumerate(steps):
            if s.get("exitCode", 0) != 0 or s.get("failureEvidence"):
                first_fail_idx = i
                break

        if first_fail_idx == -1:
            for i, s in enumerate(steps):
                out = str(s.get("output", "")).lower()
                if "error" in out or "fail" in out or "invariant violation" in out:
                    first_fail_idx = i
                    break

        # If no failure detected or failure is terminal with no recovery, cannot harvest contrastive pair
        if first_fail_idx == -1 or first_fail_idx == len(steps) - 1:
            return [], []

        # Identify fork step: if the failing step was a test/command and the preceding step was an edit/patch,
        # the divergent hypothesis started at the edit.
        fork_candidate = first_fail_idx
        if first_fail_idx > 0 and steps[first_fail_idx].get("tool") in ["bash", "test_runner", "pytest", "cmd"]:
            prev_tool = steps[first_fail_idx - 1].get("tool", "")
            if prev_tool in ["edit", "write", "patch", "str_replace_editor", "bash"]:
                fork_candidate = first_fail_idx - 1

        common_prefix = steps[:fork_candidate]
        failed_branch = steps[:first_fail_idx + 1]
        success_branch = common_prefix + steps[first_fail_idx + 1:]

        return failed_branch, success_branch

    def harvest(self, session_data: Dict[str, Any], auto_persist: bool = True) -> Optional[Dict[str, Any]]:
        """
        Extracts a ContrastiveEpisode from a completed session execution trace.
        Supports explicit failedBranch, linear chronological steps partitioning,
        and backtrackHistory synthesis.
        """
        verif = session_data.get("verification", {})
        tests_passed = verif.get("testsPassed", 0)
        tests_failed = verif.get("testsFailed", 0)

        # Invariant: Must have verified successful resolution
        if tests_passed <= 0 or tests_failed != 0:
            return None

        failed_branch = session_data.get("failedBranch", [])
        success_branch = session_data.get("successBranch", [])

        # Fallback 1: Autonomous partitioning from linear steps array
        if not failed_branch:
            steps = session_data.get("steps", [])
            if steps:
                f_part, s_part = self.partition_linear_steps(steps)
                if f_part and s_part:
                    failed_branch = f_part
                    if not success_branch:
                        success_branch = s_part

        # Fallback 2: Synthesize from backtrack history
        if not failed_branch:
            backtrack_history = session_data.get("backtrackHistory", [])
            if backtrack_history:
                # If backtrack_history starts after step 0, prepend common prefix from success_branch or steps
                if success_branch and backtrack_history[0].get("step", 0) > 0:
                    prefix_len = backtrack_history[0].get("step", 0)
                    failed_branch = success_branch[:prefix_len] + backtrack_history
                elif session_data.get("steps") and backtrack_history[0].get("step", 0) > 0:
                    prefix_len = backtrack_history[0].get("step", 0)
                    failed_branch = session_data["steps"][:prefix_len] + backtrack_history
                else:
                    failed_branch = backtrack_history

        # Fallback 3: If successBranch is not provided but steps exists
        if not success_branch and session_data.get("steps"):
            success_branch = session_data.get("steps")

        # If still no failed branch, cannot construct a contrastive episode
        if not failed_branch or not success_branch:
            return None

        fork_idx = self.find_lowest_common_ancestor(failed_branch, success_branch)
        pivot_delta = self.compute_causal_pivot_delta(failed_branch, success_branch, fork_idx)

        init_state = session_data.get("initialState", {})
        diag_error = init_state.get("diagnosticError", "")
        if not diag_error and failed_branch:
            failed_ev = failed_branch[-1].get("failureEvidence", {})
            diag_error = failed_ev.get("rootMessage", "") or str(failed_branch[-1].get("output", ""))[:200]
            init_state["diagnosticError"] = diag_error

        lesson = session_data.get("lesson")
        if not lesson:
            lesson = self.synthesize_lesson(diag_error, failed_branch, success_branch, pivot_delta)

        # Construct ContrastiveEpisode object
        now_iso = datetime.now(timezone.utc).isoformat()
        scope = session_data.get("scope", {
            "repository": "workspace_repo",
            "subsystem": "core",
            "language": "python",
            "taskType": "compilation_fix"
        })
        prov = session_data.get("provenance", {
            "originSessionId": session_data.get("sessionId", "sess_unknown"),
            "originRunId": session_data.get("runId", "run_unknown"),
            "modelId": "zhipu/glm-5.3-flash",
            "piRevision": "11894012dd461232eb075bc890538b6866860a10"
        })

        failed_traj_actions = [{"tool": s.get("tool", "cmd"), "args": s.get("args", {})} for s in failed_branch]
        failed_traj_obs = [{"exitCode": s.get("exitCode", 1), "outputSnippet": s.get("output", "")} for s in failed_branch]
        succ_traj_actions = [{"tool": s.get("tool", "cmd"), "args": s.get("args", {})} for s in success_branch]
        succ_traj_obs = [{"exitCode": s.get("exitCode", 0), "outputSnippet": s.get("output", "")} for s in success_branch]

        episode = {
            "episodeId": f"ep_harvest_{hashlib.md5((session_data.get('sessionId', '') + now_iso).encode()).hexdigest()[:10]}",
            "schemaVersion": "1.0.0",
            "problemSignature": session_data.get("problemSignature") or ContrastiveMemoryStore.hash_diagnostic(diag_error),
            "provenance": prov,
            "scope": scope,
            "initialState": init_state,
            "failedTrajectory": {
                "hypothesis": failed_branch[0].get("hypothesis", "Initial hypothesis"),
                "actions": failed_traj_actions,
                "observations": failed_traj_obs,
                "failureEvidence": failed_branch[-1].get("failureEvidence", {"errorCategory": "runtime", "rootMessage": "Execution error"})
            },
            "successfulTrajectory": {
                "hypothesis": success_branch[0].get("hypothesis", "Pivoted resolution"),
                "actions": succ_traj_actions,
                "observations": succ_traj_obs,
                "successEvidence": {"testsPassed": tests_passed, "testsFailed": tests_failed, "outputSnippet": "All tests green."}
            },
            "divergencePoint": pivot_delta,
            "lesson": lesson,
            "verification": {
                "testsPassed": tests_passed,
                "testsFailed": tests_failed
            }
        }

        if auto_persist:
            ep_id = self.store.write(episode)
            episode["episodeId"] = ep_id

        return episode


if __name__ == "__main__":
    harvester = ContrastiveHarvester(ContrastiveMemoryStore(":memory:"))
    print("ContrastiveHarvester initialized successfully.")
