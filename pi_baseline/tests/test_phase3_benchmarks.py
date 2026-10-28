#!/usr/bin/env python3
"""
Comprehensive Phase 3 Verification Suite:
Tests:
  1. Task Suite Synthesis & Stratification (40 tasks, 20-train/20-eval, Python/SQL, Easy/Med/Hard).
  2. Anti-Contamination Static Audit (0% literal code overlap, 0% test assertion overlap).
  3. Unidirectional Memory Access Guard (T_train: Write / T_test: Read-Only via ReadOnlyMemoryProxy).
  4. Option B Harvested Episodes Integrity (20 active contrastive episodes in SQLite with FTS5).
  5. Preliminary 3-Arm Comparative Baseline Metrics (Arm 0 vs Arm 1 vs Arm 3 telemetry verification).
"""

import json
import os
import sqlite3
import sys
from pathlib import Path

# Path setup
TESTS_DIR = Path(__file__).resolve().parent
BASELINE_DIR = TESTS_DIR.parent
HARNESS_DIR = BASELINE_DIR / "harness"
sys.path.insert(0, str(HARNESS_DIR))
sys.path.insert(0, str(HARNESS_DIR / "memory"))
sys.path.insert(0, str(HARNESS_DIR / "hooks"))

from benchmark_loader import BenchmarkLoader, MemoryContaminationError, ReadOnlyMemoryProxy
from contrastive_store import ContrastiveMemoryStore


def test_1_task_suite_synthesis_and_stratification():
    """Validates 40 tasks with exact 20/20 train/test split and tier stratification."""
    bench_dir = BASELINE_DIR / "tasks" / "benchmarks"
    train_file = bench_dir / "train_stream_v1.json"
    eval_file = bench_dir / "eval_stream_v1.json"
    manifest_file = bench_dir / "benchmark_manifest.json"

    assert train_file.exists(), f"Missing {train_file}"
    assert eval_file.exists(), f"Missing {eval_file}"
    assert manifest_file.exists(), f"Missing {manifest_file}"

    with open(train_file, "r", encoding="utf-8") as f:
        train_tasks = json.load(f)
    with open(eval_file, "r", encoding="utf-8") as f:
        eval_tasks = json.load(f)
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert len(train_tasks) == 20, f"Expected 20 train tasks, got {len(train_tasks)}"
    assert len(eval_tasks) == 20, f"Expected 20 eval tasks, got {len(eval_tasks)}"
    assert manifest["total_tasks"] == 40

    required_keys = {
        "task_id", "domain", "split", "complexity", "title",
        "instruction", "initial_code", "test_code", "solution_patch",
        "trap_indicator", "counterfactual_pivot"
    }

    for split_name, task_list in [("train", train_tasks), ("eval", eval_tasks)]:
        py_tasks = [t for t in task_list if t["domain"] == "python"]
        sql_tasks = [t for t in task_list if t["domain"] == "sql"]
        assert len(py_tasks) == 10, f"Expected 10 Python tasks in {split_name}, got {len(py_tasks)}"
        assert len(sql_tasks) == 10, f"Expected 10 SQL tasks in {split_name}, got {len(sql_tasks)}"

        for domain_name, d_tasks in [("python", py_tasks), ("sql", sql_tasks)]:
            easy = [t for t in d_tasks if t["complexity"] == "easy"]
            med = [t for t in d_tasks if t["complexity"] == "medium"]
            hard = [t for t in d_tasks if t["complexity"] == "hard"]
            assert len(easy) == 3, f"Expected 3 Easy {domain_name} in {split_name}, got {len(easy)}"
            assert len(med) == 4, f"Expected 4 Medium {domain_name} in {split_name}, got {len(med)}"
            assert len(hard) == 3, f"Expected 3 Hard {domain_name} in {split_name}, got {len(hard)}"

        for t in task_list:
            missing = required_keys - set(t.keys())
            assert not missing, f"Task {t.get('task_id')} missing keys: {missing}"

    print("PASS: test_1_task_suite_synthesis_and_stratification")


def test_2_anti_contamination_audit():
    """Validates 0% literal code and 0% test assertion overlap between train and eval splits."""
    loader = BenchmarkLoader()
    report = loader.run_anti_contamination_audit()
    assert report["audit_passed"] is True, f"Anti-contamination audit failed: {report}"
    assert report["id_collisions"] == 0
    assert report["initial_code_collisions"] == 0
    assert report["solution_patch_collisions"] == 0
    assert report["assertion_collisions"] == 0
    assert report["literal_code_overlap_rate"] == 0.0
    assert report["assertion_overlap_rate"] == 0.0
    print("PASS: test_2_anti_contamination_audit")


def test_3_unidirectional_memory_access_guard():
    """Validates that evaluation split strictly enforces read-only episodic memory access."""
    loader = BenchmarkLoader()

    class MockMemoryStore:
        def __init__(self):
            self.write_called = False
        def retrieve(self, *args, **kwargs):
            return [{"episode_id": "ep_test", "lesson": "test"}]
        def write(self, *args, **kwargs):
            self.write_called = True
            return "ep_new"
        def revise(self, *args, **kwargs):
            return "ep_revised"
        def forget(self, *args, **kwargs):
            return True

    raw_store = MockMemoryStore()

    # 1. Train split permits writes
    train_store = loader.wrap_store_for_split(raw_store, "train")
    res = train_store.write({"dummy": 1})
    assert res == "ep_new"
    assert raw_store.write_called is True

    # 2. Eval split blocks writes
    eval_store = loader.wrap_store_for_split(raw_store, "eval")
    assert isinstance(eval_store, ReadOnlyMemoryProxy)
    retrieved = eval_store.retrieve("query")
    assert len(retrieved) == 1

    try:
        eval_store.write({"leak": "data"})
        assert False, "ReadOnlyMemoryProxy failed to block write()!"
    except MemoryContaminationError:
        pass

    try:
        eval_store.revise("ep_test", {"new": 1})
        assert False, "ReadOnlyMemoryProxy failed to block revise()!"
    except MemoryContaminationError:
        pass

    try:
        eval_store.forget("ep_test")
        assert False, "ReadOnlyMemoryProxy failed to block forget()!"
    except MemoryContaminationError:
        pass

    print("PASS: test_3_unidirectional_memory_access_guard")


def test_4_option_b_harvested_episodes_integrity():
    """Validates that the SQLite database has 20 active contrastive episodes with FTS5 search."""
    db_path = BASELINE_DIR / "episodic_memory.sqlite"
    assert db_path.exists(), f"SQLite database {db_path} does not exist"

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # 1. Count active episodes
    count = cur.execute("SELECT count(*) FROM episodes WHERE is_tombstoned = 0;").fetchone()[0]
    assert count >= 20, f"Expected at least 20 active episodes, found {count}"

    # 2. Schema integrity
    rows = cur.execute("""
        SELECT episode_id, problem_signature, language, subsystem, diagnostic_error,
               lesson_invariant, lesson_prescription, initial_state_json
        FROM episodes
        WHERE is_tombstoned = 0;
    """).fetchall()

    for r in rows:
        ep_id, sig, lang, sub, diag, inv, presc, init_json = r
        assert ep_id.startswith("ep_harvest_")
        assert len(sig) > 0
        assert lang in ("python", "sql")
        assert len(diag) > 0
        assert len(inv) > 0
        assert len(presc) > 0
        init_data = json.loads(init_json)
        assert "diagnosticError" in init_data

    # 3. Test FTS5 search on lessons
    fts_rows = cur.execute("""
        SELECT episodes.episode_id, episodes.lesson_invariant
        FROM episodes_fts
        JOIN episodes ON episodes_fts.rowid = episodes.rowid
        WHERE episodes_fts MATCH 'pivot'
        LIMIT 5;
    """).fetchall()
    assert len(fts_rows) > 0, "FTS5 query failed to return matches for 'pivot'"

    conn.close()
    print("PASS: test_4_option_b_harvested_episodes_integrity")


def test_5_preliminary_3arm_baseline_metrics():
    """Validates the preliminary comparison run artifacts and metrics summary."""
    summary_file = BASELINE_DIR / "runs" / "preliminary_comparison_summary.json"
    assert summary_file.exists(), f"Summary file {summary_file} does not exist"

    with open(summary_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    expected_arms = ["arm_0_no_memory", "arm_1_ordinary_rag", "arm_3_contrastive_memory"]
    for arm in expected_arms:
        assert arm in metrics, f"Missing arm {arm} in metrics summary"
        arm_data = metrics[arm]
        assert arm_data["total_tasks"] == 6, f"Expected 6 tasks for {arm}, got {arm_data['total_tasks']}"
        assert arm_data["pass_rate_pct"] == 100.0
        assert arm_data["trap_avoidance_rate_pct"] == 100.0
        assert len(arm_data["task_breakdown"]) == 6

    # Verify run directories exist for all 3 arms
    runs_dir = BASELINE_DIR / "runs"
    for arm in expected_arms:
        arm_dir = runs_dir / arm
        assert arm_dir.exists(), f"Run directory {arm_dir} does not exist"
        runs = list(arm_dir.glob("*_seed_42"))
        assert len(runs) >= 6, f"Expected at least 6 run directories in {arm_dir}, got {len(runs)}"

    print("PASS: test_5_preliminary_3arm_baseline_metrics")


if __name__ == "__main__":
    test_1_task_suite_synthesis_and_stratification()
    test_2_anti_contamination_audit()
    test_3_unidirectional_memory_access_guard()
    test_4_option_b_harvested_episodes_integrity()
    test_5_preliminary_3arm_baseline_metrics()
    print("\nALL 5 PHASE 3 BENCHMARK VERIFICATION TESTS PASSED SUCCESSFULLY!")
