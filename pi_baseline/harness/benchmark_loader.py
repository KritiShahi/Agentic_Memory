#!/usr/bin/env python3
"""
Benchmark Loader & Anti-Contamination Audit Engine for Phase 3.
Enforces strict anti-contamination standards, loads stratified task streams,
and provides unidirectional memory access guarantees (T_train: Write / T_test: Read-Only).
"""

import json
import re
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple


class MemoryContaminationError(RuntimeError):
    """Raised when an evaluation task attempts to mutate the episodic memory store."""
    pass


class ReadOnlyMemoryProxy:
    """
    Transparent proxy wrapping ContrastiveMemoryStore during evaluation runs.
    Permits retrieve() operations while strictly blocking write(), revise(), and forget().
    """
    def __init__(self, target_store):
        self._target_store = target_store

    def retrieve(self, *args, **kwargs):
        return self._target_store.retrieve(*args, **kwargs)

    def write(self, *args, **kwargs):
        raise MemoryContaminationError(
            "Evaluation invariant violation: Attempted write() operation during held-out evaluation phase!"
        )

    def revise(self, *args, **kwargs):
        raise MemoryContaminationError(
            "Evaluation invariant violation: Attempted revise() operation during held-out evaluation phase!"
        )

    def forget(self, *args, **kwargs):
        raise MemoryContaminationError(
            "Evaluation invariant violation: Attempted forget() operation during held-out evaluation phase!"
        )

    def __getattr__(self, name: str):
        return getattr(self._target_store, name)


class BenchmarkLoader:
    """
    Loads Phase 3 benchmark datasets and performs rigorous anti-contamination audits.
    """
    def __init__(self, benchmarks_dir: Optional[str] = None):
        if benchmarks_dir is None:
            self.benchmarks_dir = Path(__file__).resolve().parent.parent / "tasks" / "benchmarks"
        else:
            self.benchmarks_dir = Path(benchmarks_dir)

        self.train_file = self.benchmarks_dir / "train_stream_v1.json"
        self.eval_file = self.benchmarks_dir / "eval_stream_v1.json"
        self.manifest_file = self.benchmarks_dir / "benchmark_manifest.json"

        self._train_tasks: List[Dict[str, Any]] = []
        self._eval_tasks: List[Dict[str, Any]] = []
        self._manifest: Dict[str, Any] = {}
        self._load_datasets()

    def _load_datasets(self):
        if not self.train_file.exists():
            raise FileNotFoundError(f"Missing train stream file: {self.train_file}")
        if not self.eval_file.exists():
            raise FileNotFoundError(f"Missing eval stream file: {self.eval_file}")
        if not self.manifest_file.exists():
            raise FileNotFoundError(f"Missing manifest file: {self.manifest_file}")

        with open(self.train_file, "r", encoding="utf-8") as f:
            self._train_tasks = json.load(f)
        with open(self.eval_file, "r", encoding="utf-8") as f:
            self._eval_tasks = json.load(f)
        with open(self.manifest_file, "r", encoding="utf-8") as f:
            self._manifest = json.load(f)

    def get_train_tasks(
        self,
        domain: Optional[str] = None,
        complexity: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        tasks = self._train_tasks
        if domain:
            tasks = [t for t in tasks if t["domain"].lower() == domain.lower()]
        if complexity:
            tasks = [t for t in tasks if t["complexity"].lower() == complexity.lower()]
        return tasks

    def get_eval_tasks(
        self,
        domain: Optional[str] = None,
        complexity: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        tasks = self._eval_tasks
        if domain:
            tasks = [t for t in tasks if t["domain"].lower() == domain.lower()]
        if complexity:
            tasks = [t for t in tasks if t["complexity"].lower() == complexity.lower()]
        return tasks

    def get_representative_subset(self, per_tier: int = 1) -> List[Dict[str, Any]]:
        """
        Extracts a balanced representative evaluation subset:
        per_tier tasks per (domain, complexity) pair.
        Default (per_tier=1) produces 6 tasks:
          - Python: 1 Easy, 1 Medium, 1 Hard
          - SQL: 1 Easy, 1 Medium, 1 Hard
        """
        selected = []
        for domain in ["python", "sql"]:
            for complexity in ["easy", "medium", "hard"]:
                candidates = self.get_eval_tasks(domain=domain, complexity=complexity)
                selected.extend(candidates[:per_tier])
        return selected

    def wrap_store_for_split(self, store, split: str):
        """
        Enforces unidirectional memory permissions:
        - 'train': returns native store allowing writes/harvesting.
        - 'eval' or 'test': wraps store in ReadOnlyMemoryProxy.
        """
        if split.lower() in ("eval", "test"):
            return ReadOnlyMemoryProxy(store)
        return store

    def run_anti_contamination_audit(self) -> Dict[str, Any]:
        """
        Executes comprehensive static and lexical anti-contamination assertions:
        1. 0% task ID collision.
        2. 0% literal code hash collision (initial_code & solution_patch).
        3. 0% test assertion overlap.
        4. Domain & complexity matrix parity.
        """
        train_ids = {t["task_id"] for t in self._train_tasks}
        eval_ids = {t["task_id"] for t in self._eval_tasks}
        id_collision = len(train_ids.intersection(eval_ids))

        # 1. Literal code hashes
        train_init_hashes = {hashlib.sha256(t["initial_code"].strip().encode()).hexdigest() for t in self._train_tasks}
        eval_init_hashes = {hashlib.sha256(t["initial_code"].strip().encode()).hexdigest() for t in self._eval_tasks}
        init_code_collision = len(train_init_hashes.intersection(eval_init_hashes))

        train_patch_hashes = {hashlib.sha256(t["solution_patch"].strip().encode()).hexdigest() for t in self._train_tasks}
        eval_patch_hashes = {hashlib.sha256(t["solution_patch"].strip().encode()).hexdigest() for t in self._eval_tasks}
        patch_code_collision = len(train_patch_hashes.intersection(eval_patch_hashes))

        # 2. Extract normalized assertions
        def extract_assertions(test_code: str) -> Set[str]:
            lines = test_code.splitlines()
            assertions = set()
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("assert "):
                    # Normalize whitespace and quotes
                    norm = re.sub(r"\s+", " ", stripped)
                    assertions.add(norm)
            return assertions

        train_assertions: Set[str] = set()
        for t in self._train_tasks:
            train_assertions.update(extract_assertions(t["test_code"]))

        eval_assertions: Set[str] = set()
        for t in self._eval_tasks:
            eval_assertions.update(extract_assertions(t["test_code"]))

        assertion_collision = len(train_assertions.intersection(eval_assertions))

        is_clean = (
            id_collision == 0
            and init_code_collision == 0
            and patch_code_collision == 0
            and assertion_collision == 0
        )

        report = {
            "audit_passed": is_clean,
            "train_task_count": len(self._train_tasks),
            "eval_task_count": len(self._eval_tasks),
            "id_collisions": id_collision,
            "initial_code_collisions": init_code_collision,
            "solution_patch_collisions": patch_code_collision,
            "assertion_collisions": assertion_collision,
            "unique_train_assertions": len(train_assertions),
            "unique_eval_assertions": len(eval_assertions),
            "literal_code_overlap_rate": 0.0 if (init_code_collision + patch_code_collision) == 0 else 1.0,
            "assertion_overlap_rate": 0.0 if assertion_collision == 0 else (assertion_collision / len(eval_assertions))
        }

        if not is_clean:
            raise AssertionError(f"Anti-contamination audit failed! Report: {report}")

        return report


if __name__ == "__main__":
    loader = BenchmarkLoader()
    print("Executing BenchmarkLoader Self-Audit...")
    audit_report = loader.run_anti_contamination_audit()
    print("Anti-Contamination Audit Summary:")
    print(json.dumps(audit_report, indent=2))

    probe = loader.get_representative_subset(per_tier=1)
    print(f"\nExtracted {len(probe)}-Task Representative Evaluation Probe:")
    for task in probe:
        print(f"  • [{task['domain'].upper():6s} | {task['complexity'].upper():6s}] {task['task_id']} — {task['title']}")

    # Test Read-Only Memory Proxy guard
    class MockStore:
        def retrieve(self, q): return ["episode_1"]
        def write(self, data): return "ep_123"

    raw_store = MockStore()
    eval_store = loader.wrap_store_for_split(raw_store, "eval")
    assert eval_store.retrieve("query") == ["episode_1"]
    try:
        eval_store.write({"test": "data"})
        print("ERROR: Read-only guard failed to block write!")
    except MemoryContaminationError as e:
        print(f"\nMemory Guard Verification: Successfully blocked write in eval split -> {e}")
