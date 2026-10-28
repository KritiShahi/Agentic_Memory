#!/usr/bin/env python3
"""
Contrastive Verifier & Ranker for Candidate Patches.
Scores candidate patches based on:
1. Surgical Diff Parsimony (Occam's razor: penalizes sprawling or unnecessary edits)
2. Contrastive Invariant Compliance (penalizes blanket exception silencing, deleted tests)
3. Defensive Guard Quality (rewards boundary checks, None guards, type coercions)
"""

import ast
import re
from typing import Dict, Any, List, Tuple

class ContrastiveVerifier:
    def __init__(self):
        # Anti-pattern regexes to detect cheap bypasses
        self.anti_patterns = [
            (re.compile(r"except\s*:\s*(pass|return\b)", re.IGNORECASE), "blanket_except_pass", -10.0),
            (re.compile(r"except\s+Exception\s*:\s*(pass|return\b)", re.IGNORECASE), "blanket_exception_pass", -8.0),
            (re.compile(r"try\s*:.*?except", re.DOTALL), "try_except_wrapper", -2.0),
            (re.compile(r"^\s*-\s*assert\b", re.MULTILINE), "deleted_assertion", -6.0),
            (re.compile(r"^\s*\+\s*return\s+(True|False|None)\s*$", re.MULTILINE), "trivial_return_bypass", -3.0),
            (re.compile(r"time\.sleep\s*\(", re.IGNORECASE), "sleep_delay_anti_pattern", -8.0),
        ]
        
        # Positive invariant patterns
        self.positive_patterns = [
            (re.compile(r"if\s+[\w\.]+\s+is\s+None\b"), "none_guard", 2.0),
            (re.compile(r"if\s+not\s+[\w\.]+"), "not_empty_guard", 1.5),
            (re.compile(r"isinstance\s*\("), "type_guard", 2.0),
            (re.compile(r"len\s*\([\w\.]+\)\s*(==|>|<|>=|<=)\s*0"), "boundary_guard", 2.0),
            (re.compile(r"getattr\s*\("), "safe_attribute_access", 1.5),
        ]

    def score_patch(self, patch_diff: str, parsed_ast: bool = True) -> Dict[str, Any]:
        """Calculates a composite verification score for a candidate patch."""
        if not patch_diff or not parsed_ast:
            return {
                "total_score": -999.0,
                "parsimony_score": 0.0,
                "invariant_score": -999.0,
                "flags": ["invalid_or_empty"],
                "lines_added": 0,
                "lines_deleted": 0
            }
            
        lines_added = sum(1 for line in patch_diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
        lines_deleted = sum(1 for line in patch_diff.splitlines() if line.startswith("-") and not line.startswith("---"))
        total_changed = lines_added + lines_deleted
        
        # Parsimony score: 1.0 for small surgical edits (<= 10 lines), decaying for bloated edits
        parsimony_score = 1.0 / (1.0 + 0.05 * max(0, total_changed - 5))
        
        # Invariant score
        invariant_score = 5.0 # Base score for compiling code
        flags = []
        
        for pattern, name, weight in self.anti_patterns:
            if pattern.search(patch_diff):
                invariant_score += weight
                flags.append(f"PENALTY_{name}")
                
        for pattern, name, weight in self.positive_patterns:
            if pattern.search(patch_diff):
                invariant_score += weight
                flags.append(f"BONUS_{name}")
                
        total_score = parsimony_score * 5.0 + invariant_score
        
        return {
            "total_score": round(total_score, 3),
            "parsimony_score": round(parsimony_score, 3),
            "invariant_score": round(invariant_score, 3),
            "flags": flags,
            "lines_added": lines_added,
            "lines_deleted": lines_deleted,
            "total_lines_changed": total_changed
        }

    def rank_candidates(self, candidates: List[Dict[str, Any]]) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Ranks candidate patches.
        Input: list of dicts with {"patch": str, "test_passed": bool, "ast_passed": bool}
        Returns: list of (candidate_index, score, score_details) sorted descending by score.
        """
        scored = []
        for idx, cand in enumerate(candidates):
            patch = cand.get("patch", "")
            ast_ok = cand.get("ast_passed", True)
            test_ok = cand.get("test_passed", False)
            
            score_data = self.score_patch(patch, parsed_ast=ast_ok)
            # Heavy penalty if test oracle failed
            if not test_ok:
                score_data["total_score"] -= 100.0
                score_data["flags"].append("TEST_ORACLE_FAILED")
            else:
                score_data["total_score"] += 50.0
                score_data["flags"].append("TEST_ORACLE_PASSED")
                
            scored.append((idx, score_data["total_score"], score_data))
            
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
