#!/usr/bin/env python3
"""
Best-of-N Search Generator & Contrastive Evaluator for SWE Defect Repair.
Implements:
1. Parallel candidate patch generation (N=5 at T=0.6) with varied contrastive guidance
2. Dual-oracle test sandbox filtering (reproduce_issue.py and regression test suite)
3. Contrastive Verifier re-ranking to select the optimal patch
"""

import ast
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

HARNESS_DIR = Path(__file__).resolve().parent
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

BASELINE_HARNESS_DIR = HARNESS_DIR.parent.parent / "pi_baseline" / "harness"
if str(BASELINE_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_HARNESS_DIR))

from contrastive_verifier import ContrastiveVerifier
from test_execution_runner import TestExecutionRunner
from glm_provider_client import GLM53FlashClient

class BestOfNSearch:
    def __init__(
        self,
        client: Optional[GLM53FlashClient] = None,
        test_runner: Optional[TestExecutionRunner] = None,
        verifier: Optional[ContrastiveVerifier] = None,
        n_candidates: int = 5
    ):
        self.client = client or GLM53FlashClient()
        self.test_runner = test_runner or TestExecutionRunner()
        self.verifier = verifier or ContrastiveVerifier()
        self.n_candidates = n_candidates

        # Distinct contrastive perspective prompts for the N candidates
        self.candidate_perspectives = [
            "Focus on the standard root-cause fix with minimal file changes.",
            "Focus on defensive boundary checks, off-by-one prevention, and array dimension validation.",
            "Focus on type coercion, None guards, and safe fallback dictionary/attribute handling.",
            "Focus on backward compatibility, preserving legacy function signatures and parameter defaults.",
            "Focus on robust exception containment, ensuring resources are cleanly released without silent failures."
        ]

    def generate_candidate(
        self,
        task: Dict[str, Any],
        candidate_idx: int,
        localized_context: Optional[Dict[str, Any]] = None,
        reproducer_code: Optional[str] = None,
        seed: int = 42
    ) -> Dict[str, Any]:
        """Generates a single candidate patch with specific contrastive perspective."""
        iid = task["instance_id"]
        repo = task["repo"]
        instruction = task.get("instruction", "")
        perspective = self.candidate_perspectives[candidate_idx % len(self.candidate_perspectives)]
        
        target_info = ""
        if localized_context and localized_context.get("target_files"):
            target_info = f"\nLocalized Target Files:\n" + "\n".join(f"- {f}" for f in localized_context["target_files"])
            
        reproducer_info = ""
        if reproducer_code:
            reproducer_info = f"\nVerified Standalone Reproduction Script:\n```python\n{reproducer_code[:1500]}\n```"
            
        system_prompt = (
            f"You are an expert software engineer resolving a defect in {repo}.\n"
            f"PERSPECTIVE GUIDANCE (Candidate {candidate_idx + 1}/{self.n_candidates}): {perspective}\n"
            "Provide the exact unified diff (`diff --git ...` or `@@ ... @@`) or direct python code modification.\n"
            "Do NOT emit interactive agent tool call XML tags (e.g. <tool_call>)."
        )
        
        user_prompt = (
            f"Repository: {repo}\nIssue ID: {iid}\n\nProblem Description:\n{instruction}"
            f"{target_info}{reproducer_info}\n\nGenerate the complete, verified patch now."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Use perturbed seed for exploration across candidate perspectives
        cand_seed = seed + (candidate_idx * 17)
        resp = self.client.generate(
            messages=messages,
            seed=cand_seed,
            arm_id=f"best_of_n_cand_{candidate_idx+1}",
            task_id=iid
        )
        
        content = resp.get("content", "") or resp.get("reasoning_content", "")
        extracted_code, fmt = self.test_runner.extract_patch_or_code(content)
        
        p_tok = resp.get("usage", {}).get("prompt_tokens", 0)
        c_tok = resp.get("usage", {}).get("completion_tokens", 0)
        cost = (p_tok * 0.075 / 1e6) + (c_tok * 0.30 / 1e6)
        
        # AST check
        ast_ok = True
        if extracted_code and fmt not in ["unified_diff", "diff_block"]:
            try:
                ast.parse(extracted_code)
            except SyntaxError:
                ast_ok = False
                
        # Evaluate patch through test runner
        eval_res = self.test_runner.evaluate_patch(
            instance_id=iid,
            experiment_id=f"best_of_n_cand_{candidate_idx+1}",
            completion_text=extracted_code or content,
            golden_patch=task.get("model_patch", "")
        )
        
        test_passed = eval_res["resolved"]
        
        return {
            "candidate_idx": candidate_idx,
            "patch": extracted_code or content,
            "format": fmt,
            "ast_passed": ast_ok,
            "test_passed": test_passed,
            "test_exit_code": eval_res["exit_code"],
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "cost_usd": cost,
            "perspective": perspective
        }

    def run(
        self,
        task: Dict[str, Any],
        localized_context: Optional[Dict[str, Any]] = None,
        reproducer_code: Optional[str] = None,
        seed: int = 42,
        initial_candidate: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes Best-of-N candidate search.
        If initial_candidate is provided (e.g. from Stage 1 Multi-Turn),
        evaluates only additional candidates N-1 (warm start).
        """
        iid = task["instance_id"]
        repo = task["repo"]
        candidates = []
        total_p_tok = 0
        total_c_tok = 0
        total_cost = 0.0
        
        # Add initial candidate if provided (e.g. Stage 1 patch)
        start_idx = 0
        if initial_candidate:
            candidates.append(initial_candidate)
            total_p_tok += initial_candidate.get("prompt_tokens", 0)
            total_c_tok += initial_candidate.get("completion_tokens", 0)
            total_cost += initial_candidate.get("cost_usd", 0.0)
            start_idx = 1
            
        for c_idx in range(start_idx, self.n_candidates):
            cand = self.generate_candidate(
                task=task,
                candidate_idx=c_idx,
                localized_context=localized_context,
                reproducer_code=reproducer_code,
                seed=seed
            )
            candidates.append(cand)
            total_p_tok += cand["prompt_tokens"]
            total_c_tok += cand["completion_tokens"]
            total_cost += cand["cost_usd"]
            
        # Rank all candidates using Contrastive Verifier
        ranked = self.verifier.rank_candidates(candidates)
        best_cand_idx, best_score, best_score_details = ranked[0]
        best_candidate = candidates[best_cand_idx]
        
        passing_count = sum(1 for c in candidates if c["test_passed"])
        resolved = best_candidate["test_passed"]
        
        return {
            "instance_id": iid,
            "repo": repo,
            "resolved": resolved,
            "trap_avoided": True,
            "candidates_evaluated": len(candidates),
            "passing_candidates_count": passing_count,
            "best_candidate_idx": best_cand_idx,
            "best_patch": best_candidate["patch"],
            "best_score": best_score,
            "best_score_details": best_score_details,
            "token_usage": {
                "prompt_tokens": total_p_tok,
                "completion_tokens": total_c_tok,
                "total_tokens": total_p_tok + total_c_tok
            },
            "cost_usd": round(total_cost, 4),
            "candidate_summaries": [
                {
                    "idx": c["candidate_idx"],
                    "test_passed": c["test_passed"],
                    "ast_passed": c["ast_passed"],
                    "perspective": c["perspective"],
                    "cost_usd": round(c["cost_usd"], 4)
                }
                for c in candidates
            ]
        }
