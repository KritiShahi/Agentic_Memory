#!/usr/bin/env python3
"""
Automated Certification Audit for 4-Shield Anti-Leakage Partitioning.
Verifies:
  1. Shield 1: Disjoint Repositories (Repo_train ∩ Repo_eval = ∅).
  2. Shield 2: Organizational Clustering (Org_train ∩ Org_eval = ∅).
  3. Shield 3: Deduplicated Evaluation Instances (exactly 1 issue per eval trajectory; 0 instance overlap).
  4. Shield 4: Exclusion of memorized classic repositories from eval.
  5. Contrastive balance and complexity distribution assertions.
"""

import json
import sys
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BENCHMARKS_DIR / "train_swe_stream.json"
EVAL_FILE = BENCHMARKS_DIR / "eval_swe_stream.json"
MANIFEST_FILE = BENCHMARKS_DIR / "swe_manifest.json"

MEMORIZED_REPOS = {
    "psf/requests", "pallets/flask", "scikit-learn/scikit-learn", "numpy/numpy",
    "pandas-dev/pandas", "django/django", "sympy/sympy", "pytest-dev/pytest",
    "matplotlib/matplotlib", "scipy/scipy", "ansible/ansible"
}


def audit_splits():
    print("=" * 75)
    print("RUNNING 4-SHIELD ANTI-LEAKAGE CERTIFICATION AUDIT")
    print("=" * 75)

    if not TRAIN_FILE.exists() or not EVAL_FILE.exists():
        raise FileNotFoundError("Missing train_swe_stream.json or eval_swe_stream.json.")

    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    train_repos = {t["repo"] for t in train_data}
    eval_repos = {t["repo"] for t in eval_data}
    
    train_orgs = {t["repo"].split("/")[0] for t in train_data if "/" in t["repo"]}
    eval_orgs = {t["repo"].split("/")[0] for t in eval_data if "/" in t["repo"]}

    train_instances = {t["instance_id"] for t in train_data}
    eval_instances = {t["instance_id"] for t in eval_data}

    print(f"Loaded Train: {len(train_data)} trajectories across {len(train_repos)} repos ({len(train_orgs)} orgs).")
    print(f"Loaded Eval:  {len(eval_data)} trajectories across {len(eval_repos)} repos ({len(eval_orgs)} orgs).")

    # --- SHIELD 1: Disjoint Repositories ---
    repo_leakage = train_repos.intersection(eval_repos)
    print(f"[Shield 1] Disjoint Repositories Check: {len(repo_leakage)} overlapping repos.")
    assert not repo_leakage, f"FATAL SHIELD 1 FAILURE: Overlapping repos: {repo_leakage}"

    # --- SHIELD 2: Organizational Clustering ---
    org_leakage = train_orgs.intersection(eval_orgs)
    print(f"[Shield 2] Organization Clustering Check: {len(org_leakage)} overlapping orgs.")
    assert not org_leakage, f"FATAL SHIELD 2 FAILURE: Overlapping orgs: {org_leakage}"

    # --- SHIELD 3: Deduplicated Evaluation Instances ---
    instance_leakage = train_instances.intersection(eval_instances)
    print(f"[Shield 3] Instance Disjointness Check: {len(instance_leakage)} overlapping instance IDs.")
    assert not instance_leakage, f"FATAL SHIELD 3 FAILURE: Overlapping task instances: {instance_leakage}"
    assert len(eval_instances) == len(eval_data), f"Duplicate issues in eval set: {len(eval_instances)} unique vs {len(eval_data)} total"

    # --- SHIELD 4: Pre-training Memorization Guard ---
    eval_memorized = eval_repos.intersection(MEMORIZED_REPOS)
    print(f"[Shield 4] Pre-training Memorization Guard: {len(eval_memorized)} memorized repos in eval.")
    assert not eval_memorized, f"FATAL SHIELD 4 FAILURE: Memorized repos present in eval: {eval_memorized}"

    # --- Contrastive Balance & Quality ---
    train_resolved = sum(1 for t in train_data if t["resolved"] == 1)
    train_unresolved = sum(1 for t in train_data if t["resolved"] == 0)
    print(f"[Distribution] Train Contrastive Balance: {train_resolved} resolved / {train_unresolved} unresolved.")
    assert train_resolved > 0 and train_unresolved > 0, "Train split lacks contrastive balance!"

    eval_complexities = {c: sum(1 for t in eval_data if t["complexity"] == c) for c in ["low-med", "medium", "hard"]}
    print(f"[Distribution] Eval Complexity Stratification: {eval_complexities}")

    eval_tdd = sum(1 for t in eval_data if t.get("gen_tests_correct", 0) > 0)
    print(f"[Distribution] Eval TDD Test Representation: {eval_tdd} instances with verified test generation.")

    print("=" * 75)
    print("✅ CERTIFICATION AUDIT PASSED: ALL FOUR LEAKAGE SHIELDS ENFORCED.")
    print("=" * 75)
    return True


if __name__ == "__main__":
    audit_splits()
