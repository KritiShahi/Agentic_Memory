#!/usr/bin/env python3
"""
Comprehensive Phase 1 Verification Suite:
Tests the SQLite WAL storage adapter, hybrid ranking, CRUD operations,
token budgeting, negative utility pruning, and stale/conflicting memory handling.
"""

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

# Add harness/memory to import path
BASELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASELINE_DIR / "harness" / "memory"))

from contrastive_store import ContrastiveMemoryStore, DenseEmbeddingEngine


def get_sample_episode(
    episode_id="ep_ts_esm_01",
    diag_error="TS2835: Relative import paths need explicit file extensions in ECMAScript imports.",
    inv_constraint="Do NOT set moduleResolution: node in tsconfig.json.",
    prescription="Set moduleResolution: NodeNext and add explicit .js extensions to relative imports.",
    tests_passed=15,
    tests_failed=0,
    language="typescript",
    task_type="compilation_fix",
    env_snapshot=None
):
    if env_snapshot is None:
        env_snapshot = {"node_version": "18.16.0", "typescript_version": "5.0.4"}

    return {
        "episodeId": episode_id,
        "schemaVersion": "1.0.0",
        "provenance": {
            "originSessionId": "sess_101",
            "originRunId": "run_test_01",
            "modelId": "zhipu/glm-5.3-flash",
            "piRevision": "11894012dd461232eb075bc890538b6866860a10"
        },
        "scope": {
            "repository": "test_repo",
            "subsystem": "build_pipeline",
            "language": language,
            "taskType": task_type
        },
        "initialState": {
            "diagnosticError": diag_error,
            "relevantFiles": [{"path": "tsconfig.json", "hash": "abc1234"}],
            "environmentSnapshot": env_snapshot
        },
        "failedTrajectory": {
            "hypothesis": "Setting moduleResolution to node will allow extensionless imports.",
            "actions": [{"tool": "edit", "args": {"path": "tsconfig.json", "content": '{"moduleResolution": "node"}'}}],
            "observations": [{"exitCode": 1, "outputSnippet": "ERR_UNKNOWN_FILE_EXTENSION: Cannot find module ./utils"}],
            "failureEvidence": {"errorCategory": "module_resolution", "rootMessage": "ERR_UNKNOWN_FILE_EXTENSION"}
        },
        "successfulTrajectory": {
            "hypothesis": "Use NodeNext with explicit .js extension mappings.",
            "actions": [
                {"tool": "edit", "args": {"path": "tsconfig.json", "content": '{"moduleResolution": "NodeNext"}'}},
                {"tool": "edit", "args": {"path": "src/index.ts", "content": "import { helper } from './utils.js';"}}
            ],
            "observations": [{"exitCode": 0, "outputSnippet": "Build succeeded with 0 errors."}],
            "successEvidence": {"testsPassed": tests_passed, "testsFailed": tests_failed, "outputSnippet": "All tests green."}
        },
        "divergencePoint": {
            "forkStepIndex": 2,
            "failedDecision": "Downgraded to node resolution without extensions",
            "successfulDecision": "Configured NodeNext with explicit .js relative paths",
            "causalPivotDelta": "diff --git a/tsconfig.json\n-moduleResolution: node\n+moduleResolution: NodeNext"
        },
        "lesson": {
            "invariantConstraint": inv_constraint,
            "actionablePrescription": prescription
        },
        "verification": {
            "testsPassed": tests_passed,
            "testsFailed": tests_failed
        }
    }


def test_1_schema_and_acid():
    print("\n--- Test 1: SQLite Schema Initialization & ACID Mode ---")
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
        db_path = tmp.name

    try:
        store = ContrastiveMemoryStore(db_path)
        with store._get_connection() as conn:
            # Check WAL mode
            cur = conn.execute("PRAGMA journal_mode;")
            journal_mode = cur.fetchone()[0]
            assert journal_mode.lower() == "wal", f"Expected WAL mode, got {journal_mode}"
            print("  [OK] Journal mode confirmed as WAL.")

            # Check tables exist
            cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [r[0] for r in cur.fetchall()]
            for t in ["episodes", "episodes_fts", "revision_audit"]:
                assert t in tables, f"Missing table {t}"
            print("  [OK] Confirmed tables: episodes, episodes_fts, revision_audit.")
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_2_write_validation_and_rejection():
    print("\n--- Test 2: Write Validation & Invariant Enforcements ---")
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
        db_path = tmp.name

    try:
        store = ContrastiveMemoryStore(db_path)
        
        # 1. Valid write
        valid_ep = get_sample_episode("ep_valid")
        ep_id = store.write(valid_ep)
        assert ep_id == "ep_valid"
        print("  [OK] Valid episode written successfully.")

        # 2. Reject unverified episode (tests_failed > 0)
        invalid_ep = get_sample_episode("ep_invalid", tests_passed=10, tests_failed=2)
        try:
            store.write(invalid_ep)
            assert False, "Write should have failed for unverified episode"
        except ValueError as err:
            assert "Unverified episode" in str(err)
            print(f"  [OK] Rejected unverified episode as expected: {err}")

        # 3. Reject zero-pass episode
        zero_pass_ep = get_sample_episode("ep_zero", tests_passed=0, tests_failed=0)
        try:
            store.write(zero_pass_ep)
            assert False, "Write should have failed for zero tests passed"
        except ValueError as err:
            assert "Unverified episode" in str(err)
            print("  [OK] Rejected zero-test episode as expected.")
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_3_hybrid_retrieval_and_token_budget():
    print("\n--- Test 3: Hybrid Retrieval & Hard Token Budget Enforcements ---")
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
        db_path = tmp.name

    try:
        store = ContrastiveMemoryStore(db_path)

        # Write 3 distinct domain episodes
        store.write(get_sample_episode(
            episode_id="ep_ts",
            diag_error="TS2835: Relative import paths need explicit file extensions",
            inv_constraint="Do NOT set moduleResolution: node",
            prescription="Set moduleResolution: NodeNext and add .js extensions",
            language="typescript",
            task_type="compilation_fix"
        ))

        store.write(get_sample_episode(
            episode_id="ep_py",
            diag_error="RuntimeError: Task attached to a different loop in asyncio",
            inv_constraint="Do NOT call asyncio.get_event_loop() inside background threads",
            prescription="Pass loop explicitly or use asyncio.run() with thread-safe queues",
            language="python",
            task_type="runtime_error"
        ))

        store.write(get_sample_episode(
            episode_id="ep_rust",
            diag_error="E0502: cannot borrow `data` as mutable because it is also borrowed as immutable",
            inv_constraint="Do NOT hold immutable reference across mutating method call",
            prescription="Scope immutable borrow with block or clone underlying struct",
            language="rust",
            task_type="compilation_fix"
        ))

        # Query TypeScript error
        results = store.retrieve(
            query_text="Fix relative import paths in TypeScript",
            diagnostic_error="TS2835: Relative import paths need explicit file extensions",
            scope={"language": "typescript"},
            top_k=3,
            max_tokens=1000,
            score_threshold=0.60
        )

        assert len(results) >= 1, "Expected at least 1 retrieved result for TypeScript"
        top_match = results[0]
        assert top_match["episodeId"] == "ep_ts"
        assert top_match["hybridScore"] >= 0.70
        assert "NodeNext" in top_match["lesson"]["actionablePrescription"]
        print(f"  [OK] Correct top match: {top_match['episodeId']} (Hybrid Score: {top_match['hybridScore']})")

        # Test Token Budget Truncation (set max_tokens very small: 10 tokens)
        budget_limited = store.retrieve(
            query_text="Fix relative import paths",
            diagnostic_error="TS2835",
            max_tokens=10  # Less than single episode size
        )
        assert len(budget_limited) == 0, "Budget limit should have prevented over-budget injection"
        print("  [OK] Token budget ceiling enforced: successfully prevented prompt overflow.")

        # Test irrelevant query (should return 0 results due to threshold)
        irrelevant = store.retrieve(
            query_text="Completely unrelated cooking recipe query",
            diagnostic_error="CakeBakeError: Oven temperature too high",
            score_threshold=0.72
        )
        assert len(irrelevant) == 0, "Irrelevant query should yield 0 results"
        print("  [OK] Score threshold filtering confirmed: zero noise returned on irrelevant queries.")
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_4_revision_and_superseding_lineage():
    print("\n--- Test 4: Revision & Superseding Lineage Audit ---")
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
        db_path = tmp.name

    try:
        store = ContrastiveMemoryStore(db_path)

        # 1. Store initial episode
        store.write(get_sample_episode(
            episode_id="ep_v1",
            diag_error="ERR_MODULE_NOT_FOUND",
            prescription="Add .js extension to import."
        ))

        # 2. Revise with superior fix (v2)
        v2_data = get_sample_episode(
            episode_id="ep_v2",
            diag_error="ERR_MODULE_NOT_FOUND",
            prescription="Configure tsconfig moduleResolution NodeNext AND add package.json type: module"
        )
        store.revise(prior_episode_id="ep_v1", new_episode_data=v2_data, reason="Comprehensive dual-file resolution")

        # 3. Verify retrieval only returns v2 (v1 is superseded)
        res = store.retrieve(query_text="Fix ERR_MODULE_NOT_FOUND", score_threshold=0.50)
        returned_ids = [r["episodeId"] for r in res]
        assert "ep_v2" in returned_ids, "Active v2 episode should be returned"
        assert "ep_v1" not in returned_ids, "Superseded v1 episode should NOT be returned"
        print("  [OK] Superseded episode successfully excluded from active retrieval.")

        # 4. Verify revision audit log
        with store._get_connection() as conn:
            cur = conn.execute("SELECT * FROM revision_audit WHERE episode_id = 'ep_v1' AND action = 'SUPERSEDED'")
            audit_row = cur.fetchone()
            assert audit_row is not None
            assert "ep_v2" in audit_row["reason"]
            print("  [OK] Revision audit log accurately recorded superseding event.")
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_5_forget_and_negative_utility_pruning():
    print("\n--- Test 5: Forget (Tombstone) & Negative Utility Auto-Pruning ---")
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
        db_path = tmp.name

    try:
        store = ContrastiveMemoryStore(db_path)

        # 1. Test explicit forget / tombstoning
        store.write(get_sample_episode("ep_toxic", diag_error="ToxicError"))
        store.forget("ep_toxic", reason="Manual operator quarantine")

        res = store.retrieve("ToxicError", score_threshold=0.50)
        assert len(res) == 0, "Tombstoned episode must never be returned"
        print("  [OK] Explicit forget (tombstone) confirmed: record quarantined.")

        # 2. Test Negative Utility Pruning (feedback loop)
        store.write(get_sample_episode("ep_flaky", diag_error="FlakyDeadlockError"))
        
        # 1st failure feedback
        fb1 = store.record_feedback("ep_flaky", success=False, penalty_threshold=2)
        assert fb1["failurePenalties"] == 1
        assert fb1["isTombstoned"] is False
        print("  [OK] Failure penalty 1 recorded without tombstoning.")

        # 2nd failure feedback (hits threshold = 2)
        fb2 = store.record_feedback("ep_flaky", success=False, penalty_threshold=2)
        assert fb2["failurePenalties"] == 2
        assert fb2["isTombstoned"] is True
        print("  [OK] Negative Utility Pruning triggered: episode automatically tombstoned upon 2nd failure.")

        # Confirm retrieval excludes auto-pruned episode
        res_flaky = store.retrieve("FlakyDeadlockError", score_threshold=0.50)
        assert len(res_flaky) == 0, "Auto-pruned episode must be excluded from retrieval"
        print("  [OK] Auto-pruned episode successfully excluded from future queries.")
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_6_stale_and_conflicting_memory_resolution():
    print("\n--- Test 6: Stale & Conflicting Memory Resolution across Environment Drift ---")
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
        db_path = tmp.name

    try:
        store = ContrastiveMemoryStore(db_path)

        # Episode A: Old Node 14 CJS fix
        ep_old_cjs = get_sample_episode(
            episode_id="ep_node14_cjs",
            diag_error="ERR_REQUIRE_ESM",
            prescription="Use require() and install @types/node@14",
            env_snapshot={"node_version": "14.17.0", "module_system": "commonjs"}
        )
        store.write(ep_old_cjs)

        # Episode B: New Node 18 ESM fix
        ep_new_esm = get_sample_episode(
            episode_id="ep_node18_esm",
            diag_error="ERR_REQUIRE_ESM",
            prescription="Use dynamic import() or convert caller package to type: module",
            env_snapshot={"node_version": "18.16.0", "module_system": "module"}
        )
        store.write(ep_new_esm)

        # Current runtime is Node 18 ESM
        current_env = {"node_version": "18.16.0", "module_system": "module"}
        conflicts = store.resolve_conflicts("ERR_REQUIRE_ESM", current_env)

        assert len(conflicts) == 2, f"Expected 2 conflicting episodes, got {len(conflicts)}"
        top_resolved = conflicts[0]
        assert top_resolved["episode_id"] == "ep_node18_esm", f"Expected Node 18 ESM fix to rank #1, got {top_resolved['episode_id']}"
        print(f"  [OK] Conflicting memories resolved correctly: matching environment fix ({top_resolved['episode_id']}) ranked first.")
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


if __name__ == "__main__":
    try:
        test_1_schema_and_acid()
        test_2_write_validation_and_rejection()
        test_3_hybrid_retrieval_and_token_budget()
        test_4_revision_and_superseding_lineage()
        test_5_forget_and_negative_utility_pruning()
        test_6_stale_and_conflicting_memory_resolution()
        print("\n=======================================================")
        print("PHASE 1 MEMORY CRUD VERIFICATION COMPLETE: ALL PASS (6/6)")
        print("=======================================================")
    except Exception as exc:
        print(f"\nVERIFICATION FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
