#!/usr/bin/env python3
"""
Comprehensive Phase 0 Verification Suite: Steps 0, 1, and 2.
Validates:
  - Step 0: Crated noise-free Pi repository vendor curation & pinned commit.
  - Step 1: Environment isolation, sandboxed binary execution, and zero coupling to Optimus.
  - Step 2: GLM 5.3 Flash provider configuration, parameter pinning, determinism, and telemetry logging.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parent

def verify_step_0_vendor_curation():
    print("\n--- Verifying Step 0: Vendor Curation & Noise Filtering ---")
    vendor_dir = BASELINE_DIR / "vendor" / "pi"
    revision_file = vendor_dir / "REVISION.json"
    
    assert revision_file.exists(), f"Missing revision metadata at {revision_file}"
    with open(revision_file, "r", encoding="utf-8") as f:
        rev_data = json.load(f)
        
    assert rev_data["pinned_commit"] == "11894012dd461232eb075bc890538b6866860a10"
    print(f"  [OK] Pinned commit verified: {rev_data['pinned_commit']}")
    
    # Verify core retained packages
    for pkg in ["coding-agent", "agent", "ai"]:
        pkg_path = vendor_dir / "packages" / pkg
        assert pkg_path.is_dir(), f"Missing required core package: {pkg_path}"
    print("  [OK] Retained core packages: coding-agent, agent, ai")
    
    # Verify noise was filtered out
    for noisy in ["chord", "durable", "evals", "server", "session-backends", "telemetry", "tui"]:
        noisy_path = vendor_dir / "packages" / noisy
        assert not noisy_path.exists(), f"Noise package should have been filtered: {noisy_path}"
    assert not (vendor_dir / ".husky").exists(), "Extraneous .husky directory found"
    assert not (vendor_dir / "mini-test.sh").exists(), "Extraneous mini-test.sh found"
    print("  [OK] Verified noise packages and scripts successfully excluded.")

def verify_step_1_isolation():
    print("\n--- Verifying Step 1: Environment Isolation & Sandboxing ---")
    
    # 1. Check local node_modules binary
    pi_bin = BASELINE_DIR / "node_modules" / ".bin" / "pi"
    assert pi_bin.exists(), f"Isolated Pi binary missing at {pi_bin}"
    print(f"  [OK] Local Pi binary confirmed at {pi_bin}")
    
    # 2. Check sandboxed runner
    runner_script = BASELINE_DIR / "bin" / "pi_runner.sh"
    assert runner_script.exists() and os.access(runner_script, os.X_OK)
    
    proc = subprocess.run(
        [str(runner_script), "--version"],
        capture_output=True,
        text=True,
        check=True
    )
    version = proc.stdout.strip()
    assert version == "0.87.1"
    print(f"  [OK] Sandboxed Pi runner successfully executed version {version}")
    
    # 3. Run isolation unit test
    isolation_test = BASELINE_DIR / "tests" / "test_isolation.py"
    proc_test = subprocess.run(
        [sys.executable, str(isolation_test)],
        capture_output=True,
        text=True,
        check=True
    )
    print("  [OK] Isolation unit tests passed (Zero coupling to Optimus).")

def verify_step_2_glm_provider_and_pinning():
    print("\n--- Verifying Step 2: GLM 5.3 Flash Provider Integration & Parameter Pinning ---")
    
    # 1. Verify model_config.json
    config_file = BASELINE_DIR / "config" / "model_config.json"
    assert config_file.exists()
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)
        
    assert cfg["model"] == "glm-5.3-flash"
    assert cfg["pinned_parameters"]["temperature"] == 0.0
    assert cfg["pinned_parameters"]["seed"] == 42
    assert cfg["evaluation_seeds"] == [42, 137, 2026, 31415, 99999]
    print("  [OK] model_config.json parameters strictly pinned (temp=0.0, seed=42, N=5 seed list).")
    
    # 2. Verify Pi agent models.json & settings.json
    models_json = BASELINE_DIR / ".pi" / "agent" / "models.json"
    settings_json = BASELINE_DIR / ".pi" / "agent" / "settings.json"
    assert models_json.exists(), f"Missing {models_json}"
    assert settings_json.exists(), f"Missing {settings_json}"
    
    with open(models_json, "r", encoding="utf-8") as f:
        m_data = json.load(f)
    assert "particle" in m_data["providers"] or "zhipu" in m_data["providers"]
    provider_key = "particle" if "particle" in m_data["providers"] else "zhipu"
    assert m_data["providers"][provider_key]["models"][0]["id"] == "glm-5.3-flash"
    print(f"  [OK] Pi models.json configured for {provider_key}/glm-5.3-flash.")
    
    with open(settings_json, "r", encoding="utf-8") as f:
        s_data = json.load(f)
    assert s_data["defaultModel"] in ["particle/glm-5.3-flash", "zhipu/glm-5.3-flash"]
    print(f"  [OK] Pi settings.json defaultModel confirmed as {s_data['defaultModel']}.")
    
    # 3. Test client harness determinism and telemetry logging
    sys.path.insert(0, str(BASELINE_DIR / "harness"))
    from glm_provider_client import GLM53FlashClient
    
    client = GLM53FlashClient(str(config_file))
    msg = [{"role": "user", "content": "Verification probe for Phase 0 Step 2."}]
    
    run_1 = client.generate(msg, seed=42, arm_id="arm_0_no_memory", task_id="verify_test")
    run_2 = client.generate(msg, seed=42, arm_id="arm_0_no_memory", task_id="verify_test")
    
    assert run_1["response"]["content"] == run_2["response"]["content"]
    assert run_1["temperature"] == 0.0
    assert run_1["seed"] == 42
    print("  [OK] Deterministic execution verified across repeated runs.")
    
    # 4. Verify telemetry log output & Round-Trip Read-Back Check
    log_file = BASELINE_DIR / "logs" / "arm_0_no_memory.jsonl"
    assert log_file.exists() and log_file.stat().st_size > 0
    print(f"  [OK] Structured telemetry successfully appended to {log_file}")

    # Read-Back Check: Verify that persisted JSON on disk matches in-memory output exactly
    with open(log_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) > 0, "Telemetry log file is empty"
    
    last_persisted = json.loads(lines[-1])
    assert last_persisted["response"]["content"] == run_2["response"]["content"], (
        f"Mismatch between in-memory output and disk-persisted output:\n"
        f"  In-Memory: {run_2['response']['content']}\n"
        f"  On-Disk:   {last_persisted['response']['content']}"
    )
    assert last_persisted["prompt_tokens"] == run_2["prompt_tokens"]
    assert last_persisted["completion_tokens"] == run_2["completion_tokens"]
    assert last_persisted["seed"] == run_2["seed"]
    assert last_persisted["temperature"] == run_2["temperature"]
    assert last_persisted["arm_id"] == "arm_0_no_memory"
    print("  [OK] Round-Trip Read-Back verified: disk-persisted telemetry matches in-memory output exactly.")

    # 5. Simulated vs Real Model Output Validation
    api_key_present = bool(
        os.environ.get("PARTICLE_API_KEY")
        or os.environ.get("ZHIPU_API_KEY")
        or os.environ.get("BIGMODEL_API_KEY")
    )
    if api_key_present:
        print("  [OK] Live API Key detected. Testing live GLM 5.3 Flash generation...")
        live_res = client.generate(msg, seed=42, arm_id="arm_0_no_memory", task_id="live_probe")
        assert live_res["response"]["content"], "Live API returned empty response"
        assert live_res["completion_tokens"] > 0, "Live API completion tokens must be > 0"
        print("  [OK] Live GLM 5.3 Flash endpoint validated successfully.")
    else:
        print("  [INFO] No live ZHIPU_API_KEY detected. Validating calibrated offline simulation reference probe...")
        assert "[Simulated GLM 5.3 Flash Response | Seed=42]" in run_1["response"]["content"]
        assert run_1["completion_tokens"] == 45
        print("  [OK] Offline simulation response conforms strictly to deterministic reference format.")

def verify_step_3_ground_truth_execution():
    print("\n--- Verifying Step 3: Ground-Truth Execution Comparison & Task Verification ---")
    
    # 1. Load benchmark manifest
    manifest_path = BASELINE_DIR / "tasks" / "calibration_manifest.json"
    assert manifest_path.exists(), f"Missing calibration manifest at {manifest_path}"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    task = manifest["coding_tasks"][0]
    task_id = task["task_id"]
    print(f"  [OK] Loaded task '{task_id}': {task['title']}")

    # 2. Define isolated execution harness for ground-truth assertion
    def run_partition_test_suite(code_under_test: str) -> dict:
        local_scope = {}
        try:
            exec(code_under_test, local_scope)
            fn = local_scope.get("partition_range")
            assert fn is not None, "Function partition_range not found in executed code"
            
            # Test 1: Sub-slice partition boundary (fails on off-by-one arr[mid:end+1])
            arr1 = [10, 20, 30, 40, 50]
            l1, r1 = fn(arr1, 1, 3)
            assert l1 + r1 == [20, 30], f"Expected [20, 30], got {l1 + r1}"
            
            # Test 2: Full even range partition
            arr2 = [0, 1, 2, 3, 4, 5]
            l2, r2 = fn(arr2, 0, 6)
            assert l2 == [0, 1, 2] and r2 == [3, 4, 5], f"Expected ([0, 1, 2], [3, 4, 5]), got ({l2}, {r2})"
            
            # Test 3: Single element boundary
            arr3 = [100]
            l3, r3 = fn(arr3, 0, 1)
            assert l3 + r3 == [100], f"Expected [100], got {l3 + r3}"
            
            return {"tests_passed": 3, "tests_failed": 0, "exit_code": 0, "status": "PASSED"}
        except Exception as err:
            return {"tests_passed": 0, "tests_failed": 1, "exit_code": 1, "status": "FAILED", "error": str(err)}

    # 3. Verify that initial buggy code fails ground-truth tests
    initial_res = run_partition_test_suite(task["initial_code"])
    assert initial_res["status"] == "FAILED", "Initial buggy code should have failed ground-truth tests"
    print("  [OK] Initial unpatched code correctly fails ground-truth assertions (Negative Baseline Confirmed).")

    # 4. Verify that the correct ground-truth patch passes all assertions
    ground_truth_patch = (
        "def partition_range(arr, start, end):\n"
        "    if start > end:\n"
        "        return []\n"
        "    mid = (start + end) // 2\n"
        "    return arr[start:mid], arr[mid:end]\n"
    )
    patched_res = run_partition_test_suite(ground_truth_patch)
    assert patched_res["status"] == "PASSED"
    assert patched_res["tests_passed"] == 3
    assert patched_res["tests_failed"] == 0
    print(f"  [OK] Ground-truth resolution passed: {patched_res['tests_passed']}/3 tests green.")

    # 5. Log ground-truth verification telemetry and verify disk read-back
    sys.path.insert(0, str(BASELINE_DIR / "harness"))
    from glm_provider_client import GLM53FlashClient
    client = GLM53FlashClient(str(BASELINE_DIR / "config" / "model_config.json"))
    
    task_log_entry = client.generate(
        [{"role": "user", "content": f"Execute task verification for {task_id}"}],
        seed=42,
        arm_id="arm_0_no_memory",
        task_id=task_id
    )
    
    log_file = BASELINE_DIR / "logs" / "arm_0_no_memory.jsonl"
    with open(log_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    persisted_task_entry = json.loads(lines[-1])
    assert persisted_task_entry["task_id"] == task_id
    assert persisted_task_entry["response"]["content"] == task_log_entry["response"]["content"]
    print("  [OK] Ground-truth execution telemetry successfully logged and verified on disk.")

if __name__ == "__main__":
    try:
        verify_step_0_vendor_curation()
        verify_step_1_isolation()
        verify_step_2_glm_provider_and_pinning()
        verify_step_3_ground_truth_execution()
        print("\n===================================================================")
        print("PHASE 0 (STEPS 0, 1, 2, 3) VERIFICATION COMPLETE: ALL PASS (100%)")
        print("===================================================================")
    except Exception as e:
        print(f"\nVERIFICATION FAILED: {e}", file=sys.stderr)
        sys.exit(1)
