#!/usr/bin/env python3
"""
Comprehensive Phase 2 Verification Suite:
Tests:
  1. Fork-Point Harvesting & Causal Pivot Delta (Δpivot) extraction from session DAGs.
  2. Pre-Flight Context Injection Hook & hard token budget ceilings (B_ctx <= 1000 tokens).
  3. Mid-Flight beforeToolCall Trap Interceptor (proactive execution blocking & steering).
  4. Unified PiHookManager end-to-end multi-step recovery lifecycle.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Path setup
BASELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASELINE_DIR / "harness" / "memory"))
sys.path.insert(0, str(BASELINE_DIR / "harness" / "hooks"))

from contrastive_store import ContrastiveMemoryStore
from contrastive_harvester import ContrastiveHarvester
from context_injector import ContextInjector
from trap_interceptor import TrapInterceptor
from pi_hook_manager import PiHookManager


def get_sample_session_trace():
    """Returns a realistic Pi session trace with a failure backtrack branch and a successful branch."""
    return {
        "sessionId": "sess_esm_101",
        "runId": "run_phase2_01",
        "initialState": {
            "diagnosticError": "TS2835: Relative import paths need explicit file extensions in ECMAScript imports.",
            "relevantFiles": [{"path": "tsconfig.json", "hash": "h123"}],
            "environmentSnapshot": {"node_version": "18.16.0", "typescript_version": "5.0.4"}
        },
        "scope": {
            "repository": "esm_repo",
            "subsystem": "build",
            "language": "typescript",
            "taskType": "compilation_fix"
        },
        # Common Ancestor: Step 0 (inspect files)
        # Fork Point: Step 1 (Failed attempts tsconfig downgrade vs Success uses NodeNext)
        "failedBranch": [
            {
                "step": 0,
                "hypothesis": "Inspect compiler error",
                "tool": "read",
                "args": {"path": "tsconfig.json"},
                "output": "{\n  \"compilerOptions\": {\"module\": \"commonjs\"}\n}",
                "exitCode": 0
            },
            {
                "step": 1,
                "hypothesis": "Downgrade moduleResolution to node to bypass extension checks",
                "tool": "edit",
                "args": {"path": "tsconfig.json", "content": '{"moduleResolution": "node"}'},
                "output": "ERR_UNKNOWN_FILE_EXTENSION: Cannot find module ./utils",
                "exitCode": 1,
                "failureEvidence": {"errorCategory": "module_resolution", "rootMessage": "ERR_UNKNOWN_FILE_EXTENSION"}
            }
        ],
        "successBranch": [
            {
                "step": 0,
                "hypothesis": "Inspect compiler error",
                "tool": "read",
                "args": {"path": "tsconfig.json"},
                "output": "{\n  \"compilerOptions\": {\"module\": \"commonjs\"}\n}",
                "exitCode": 0
            },
            {
                "step": 1,
                "hypothesis": "Configure NodeNext moduleResolution and append .js extensions",
                "tool": "edit",
                "args": {"path": "tsconfig.json", "content": '{"moduleResolution": "NodeNext", "module": "NodeNext"}'},
                "output": "Updated tsconfig.json successfully",
                "exitCode": 0
            },
            {
                "step": 2,
                "hypothesis": "Add explicit .js extension in relative import",
                "tool": "edit",
                "args": {"path": "src/index.ts", "content": "import { helper } from './utils.js';"},
                "output": "Updated src/index.ts",
                "exitCode": 0
            }
        ],
        "verification": {
            "testsPassed": 24,
            "testsFailed": 0,
            "exitCode": 0
        }
    }


def test_1_fork_point_harvesting_and_causal_pivot():
    print("\n--- Test 1: Fork-Point Harvesting & Causal Pivot Extraction ---")
    store = ContrastiveMemoryStore(":memory:")
    harvester = ContrastiveHarvester(store)

    session_trace = get_sample_session_trace()
    episode = harvester.harvest(session_trace, auto_persist=True)

    assert episode is not None, "Harvester should have extracted an episode"
    assert episode["episodeId"].startswith("ep_harvest_")
    print(f"  [OK] Harvested Contrastive Episode: {episode['episodeId']}")

    # Check Fork Index
    div_point = episode["divergencePoint"]
    assert div_point["forkStepIndex"] == 1, f"Expected fork index 1, got {div_point['forkStepIndex']}"
    print(f"  [OK] Lowest Common Ancestor / Fork Step Index correctly identified: Step {div_point['forkStepIndex']}")

    # Check Causal Pivot Delta
    delta = div_point["causalPivotDelta"]
    assert "[-] Aborted Actions" in delta
    assert "[+] Pivoted Actions" in delta
    print(f"  [OK] Causal Pivot Delta (Δpivot) generated:\n       {delta.replace(chr(10), chr(10) + '       ')}")

    # Check Lesson Synthesis
    lesson = episode["lesson"]
    assert "do NOT attempt" in lesson["invariantConstraint"]
    assert "Apply proven pivot" in lesson["actionablePrescription"]
    print(f"  [OK] Synthesized Invariant Warning: {lesson['invariantConstraint']}")
    print(f"  [OK] Synthesized Actionable Fix:    {lesson['actionablePrescription']}")

    # Verify persisted in SQLite
    retrieved = store.retrieve("TS2835 Relative import paths", score_threshold=0.50)
    assert len(retrieved) == 1
    assert retrieved[0]["episodeId"] == episode["episodeId"]
    print("  [OK] Harvested episode successfully persisted and indexed in SQLite store.")


def test_2_pre_flight_context_injection_and_budget():
    print("\n--- Test 2: Pre-Flight Context Injection & Hard Token Budgeting ---")
    store = ContrastiveMemoryStore(":memory:")
    harvester = ContrastiveHarvester(store)
    harvester.harvest(get_sample_session_trace(), auto_persist=True)

    injector = ContextInjector(store, max_context_tokens=1000, score_threshold=0.60)

    initial_messages = [
        {"role": "system", "content": "You are a software engineering assistant."},
        {"role": "user", "content": "Fix module resolution in tsconfig.json"}
    ]

    augmented_messages, mem_tokens, retrieved_eps = injector.inject(
        messages=initial_messages,
        query_text="Fix relative import paths in tsconfig",
        diagnostic_error="TS2835: Relative import paths need explicit file extensions",
        scope={"language": "typescript"}
    )

    assert len(retrieved_eps) == 1, "Expected 1 retrieved episode"
    assert mem_tokens > 0 and mem_tokens <= 1000, f"Token budget violation: {mem_tokens} tokens"
    
    # Check that system prompt contains contrastive block
    system_content = augmented_messages[0]["content"]
    assert "=== [EPISODIC MEMORY: CONTRASTIVE GUIDANCE] ===" in system_content
    assert "INVARIANT WARNING (Anti-Pattern Trap)" in system_content
    assert "PROVEN PIVOT FIX" in system_content
    assert "Causal Pivot Delta (Δpivot)" in system_content
    print(f"  [OK] Pre-flight context hook successfully injected ({mem_tokens} tokens within 1000-token ceiling).")

    # Verify active trap extraction
    traps = injector.extract_active_traps(retrieved_eps)
    assert len(traps) >= 1
    assert any(t["tool"] == "edit" for t in traps)
    print(f"  [OK] Extracted {len(traps)} active negative action trap(s) for mid-flight monitoring.")


def test_3_mid_flight_trap_interception():
    print("\n--- Test 3: Mid-Flight beforeToolCall Trap Interception ---")
    interceptor = TrapInterceptor()

    sample_traps = [
        {
            "episodeId": "ep_ts_01",
            "tool": "edit",
            "args": {"path": "tsconfig.json", "content": '{"moduleResolution": "node"}'},
            "invariantWarning": "Do NOT set moduleResolution: node in ESM packages.",
            "provenFix": "Set moduleResolution: NodeNext and add .js extensions.",
            "causalPivot": "diff: -node +NodeNext"
        },
        {
            "episodeId": "ep_npm_02",
            "tool": "bash",
            "args": {"command": "npm i --save-dev @types/node@14"},
            "invariantWarning": "Do NOT install legacy types in Node 18 environments.",
            "provenFix": "Install @types/node@18."
        }
    ]
    interceptor.register_traps(sample_traps)

    # 1. Test Benign Tool Call (Should be admitted)
    admit_blocked, admit_msg, _ = interceptor.evaluate_tool_call(
        tool_name="bash",
        tool_args={"command": "npm test"}
    )
    assert not admit_blocked, "Benign tool call should NOT be blocked"
    assert admit_msg is None
    print("  [OK] Benign tool call ('npm test') admitted without interception.")

    # 2. Test Trapped Bash Command (Should be blocked)
    block_bash, bash_msg, trap_bash = interceptor.evaluate_tool_call(
        tool_name="bash",
        tool_args={"command": "npm i --save-dev @types/node@14"}
    )
    assert block_bash, "Dangerous bash command MUST be blocked"
    assert "EXECUTION BLOCKED [Episodic Trap Intercept]" in bash_msg
    assert "Do NOT install legacy types" in bash_msg
    print(f"  [OK] Dangerous bash command blocked before OS dispatch:\n       {bash_msg.splitlines()[0]}")

    # 3. Test Trapped File Edit (Should be blocked)
    block_edit, edit_msg, trap_edit = interceptor.evaluate_tool_call(
        tool_name="edit",
        tool_args={"path": "tsconfig.json", "content": '{"moduleResolution": "node"}'}
    )
    assert block_edit, "Dangerous file edit MUST be blocked"
    assert "Do NOT set moduleResolution: node" in edit_msg
    print(f"  [OK] Dangerous file edit blocked before OS dispatch.")

    # Check metrics
    metrics = interceptor.trap_avoidance_metrics
    assert metrics["traps_intercepted"] == 2
    assert metrics["total_evaluations"] == 3
    print(f"  [OK] Trap Avoidance Metrics verified: {metrics['traps_intercepted']}/{metrics['total_evaluations']} intercepted ({metrics['interception_rate']*100:.1f}% TAR).")


def test_4_unified_hook_manager_end_to_end_lifecycle():
    print("\n--- Test 4: Unified PiHookManager End-to-End Multi-Step Recovery ---")
    store = ContrastiveMemoryStore(":memory:")
    manager = PiHookManager(store, max_context_tokens=1000, score_threshold=0.60)

    # STEP A: Ingest Training Experience (Session Settled)
    session_trace = get_sample_session_trace()
    ep = manager.on_session_settled(session_trace)
    assert ep is not None
    print("  [OK] Step A: Training session settled and contrastive episode consolidated into SQLite.")

    # STEP B: New Session Starts (before_prompt Hook)
    prompt_msgs = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Fix TS2835 import resolution error in tsconfig.json"}
    ]
    augmented, mem_tokens, retrieved = manager.before_prompt(
        messages=prompt_msgs,
        query_text="Fix TS2835 import resolution error",
        diagnostic_error="TS2835: Relative import paths need explicit file extensions",
        scope={"language": "typescript"}
    )
    assert len(retrieved) == 1
    assert mem_tokens > 0
    assert manager.metrics["active_retrieved_count"] == 1
    print("  [OK] Step B: Pre-flight hook injected memory and auto-registered active traps.")

    # STEP C: In-Flight Execution - Model Attempts Known Bad Action
    blocked, mitigation_prompt, _ = manager.before_tool_call(
        tool_name="edit",
        tool_args={"path": "tsconfig.json", "content": '{"moduleResolution": "node"}'}
    )
    assert blocked is True
    assert mitigation_prompt is not None
    print("  [OK] Step C: Model trap attempt actively intercepted and redirected via mitigation prompt.")

    # STEP D: Model Pivots to Proven Resolution
    blocked_pivot, _, _ = manager.before_tool_call(
        tool_name="edit",
        tool_args={"path": "tsconfig.json", "content": '{"moduleResolution": "NodeNext"}'}
    )
    assert blocked_pivot is False
    print("  [OK] Step D: Verified pivot action admitted to OS without interception.")

    # STEP E: Session Finishes with Success - Lifecycle Feedback Hook
    feedback_res = manager.on_session_feedback(episode_id=retrieved[0]["episodeId"], success=True)
    assert feedback_res["successReinforcements"] == 1
    assert feedback_res["isTombstoned"] is False
    print("  [OK] Step E: Positive reinforcement feedback recorded in SQLite.")


def test_5_autonomous_in_flight_harvesting_and_linear_partitioning():
    print("\n--- Test 5: Autonomous In-Flight Harvesting & Linear Trace Partitioning ---")
    store = ContrastiveMemoryStore(":memory:")
    manager = PiHookManager(store, max_context_tokens=1000, score_threshold=0.60)

    # 1. Simulate an autonomous session "in the wild" without pre-packaged failedBranch
    # Step 0: Initial inspection
    manager.after_tool_call(
        tool_name="read",
        tool_args={"path": "calc.py"},
        exit_code=0,
        output="def add(a, b): return a - b",
        hypothesis="Inspect calc.py specification"
    )

    # Step 1: Model takes a flawed action in the wild that triggers an error
    step_1 = manager.after_tool_call(
        tool_name="edit",
        tool_args={"path": "calc.py", "content": "def add(a, b): return a * b"},
        exit_code=1,
        output="AssertionError: 2 + 3 != 6. Expected addition, got multiplication.",
        hypothesis="Naive multiplication heuristic",
        failure_evidence={
            "errorCategory": "assertion_error",
            "rootMessage": "AssertionError: 2 + 3 != 6"
        }
    )
    assert "failureEvidence" in step_1
    assert len(manager.in_flight_backtracks) == 1
    print("  [OK] Step 1: Flawed tool call observed and dynamically logged into in-flight backtracks.")

    # Step 2: Model self-corrects and applies working fix
    manager.after_tool_call(
        tool_name="edit",
        tool_args={"path": "calc.py", "content": "def add(a, b): return a + b"},
        exit_code=0,
        output="Modified calc.py successfully.",
        hypothesis="Counterfactual pivot: use addition operator"
    )

    # Step 3: Verify clean test run
    manager.after_tool_call(
        tool_name="bash",
        tool_args={"command": "pytest test_calc.py"},
        exit_code=0,
        output="1 passed in 0.02s",
        hypothesis="Confirm test suite passes"
    )

    # 2. Settle session WITHOUT providing failedBranch or backtrackHistory
    # PiHookManager must auto-supply in-flight steps and ContrastiveHarvester must partition autonomously
    raw_session_data = {
        "sessionId": "sess_wild_autonomous_001",
        "initialState": {"relevantFiles": [{"path": "calc.py", "hash": "h_calc"}]},
        "scope": {"repository": "math_repo", "subsystem": "arithmetic", "language": "python", "taskType": "bug_fix"},
        "verification": {"testsPassed": 1, "testsFailed": 0}
    }

    harvested_ep = manager.on_session_settled(raw_session_data, auto_persist=True)
    assert harvested_ep is not None, "Failed to autonomously harvest episode from in-flight steps"
    assert harvested_ep["divergencePoint"]["forkStepIndex"] == 1
    assert "[-] Aborted Actions" in harvested_ep["divergencePoint"]["causalPivotDelta"]
    assert "[+] Pivoted Actions" in harvested_ep["divergencePoint"]["causalPivotDelta"]
    assert "AssertionError: 2 + 3 != 6" in harvested_ep["lesson"]["invariantConstraint"]
    print("  [OK] Step 2: Autonomously partitioned linear trace, computed Delta_pivot, and persisted to SQLite.")

    # 3. Verify in-flight buffer was flushed
    assert len(manager.current_session_steps) == 0
    assert len(manager.in_flight_backtracks) == 0
    print("  [OK] Step 3: In-flight session buffers cleanly reset after settlement.")

    # 4. Verify backtrackHistory synthesis fallback
    raw_backtrack_session = {
        "sessionId": "sess_backtrack_test_002",
        "successBranch": [
            {"step": 0, "hypothesis": "read file", "tool": "read", "args": {"path": "f.py"}},
            {"step": 1, "hypothesis": "correct fix", "tool": "edit", "args": {"path": "f.py", "content": "good"}}
        ],
        "backtrackHistory": [
            {"step": 1, "hypothesis": "bad fix", "tool": "edit", "args": {"path": "f.py", "content": "bad"}, "exitCode": 1, "failureEvidence": {"errorCategory": "syntax", "rootMessage": "SyntaxError"}}
        ],
        "verification": {"testsPassed": 1, "testsFailed": 0}
    }
    bt_ep = manager.harvester.harvest(raw_backtrack_session, auto_persist=True)
    assert bt_ep is not None
    assert bt_ep["divergencePoint"]["forkStepIndex"] == 1
    print("  [OK] Step 4: BacktrackHistory alignment and prefix synthesis verified.")


if __name__ == "__main__":
    try:
        test_1_fork_point_harvesting_and_causal_pivot()
        test_2_pre_flight_context_injection_and_budget()
        test_3_mid_flight_trap_interception()
        test_4_unified_hook_manager_end_to_end_lifecycle()
        test_5_autonomous_in_flight_harvesting_and_linear_partitioning()
        print("\n===================================================================")
        print("PHASE 2 PI SUBSYSTEM HOOKS VERIFICATION COMPLETE: ALL PASS (5/5)")
        print("===================================================================")
    except Exception as exc:
        print(f"\nVERIFICATION FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
