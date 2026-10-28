#!/usr/bin/env python3
"""
Example 2: Simple Deterministic Query & Telemetry Probe.

This script demonstrates:
1. Sending a prompt through the GLM 5.3 Flash client harness.
2. Enforcing pinned parameters (temperature=0.0, seed=42, reasoning_effort="none", max_tokens=4096).
3. Verifying determinism: Calling generate() twice with seed=42 produces identical output.
4. Verifying disk persistence: Asserting that telemetry in logs/arm_0_no_memory.jsonl matches in-memory output.
"""

import json
import os
import sys
from pathlib import Path

# Paths relative to this script
EXAMPLES_DIR = Path(__file__).resolve().parent
BASELINE_DIR = EXAMPLES_DIR.parent
HARNESS_DIR = BASELINE_DIR / "harness"
sys.path.insert(0, str(HARNESS_DIR))

from glm_provider_client import GLM53FlashClient

def main():
    print("==================================================================")
    print("EXAMPLE 2: DETERMINISTIC MODEL QUERY & TELEMETRY PROBE")
    print("==================================================================")

    # 1. Initialize Client
    config_path = BASELINE_DIR / "config" / "model_config.json"
    client = GLM53FlashClient(str(config_path))
    
    print(f"\n1. Initialized GLM Client with configuration:")
    print(f"   - Model:            {client.model}")
    print(f"   - Endpoint:         {client.endpoint_url}")
    print(f"   - Temperature:      {client.pinned_params.get('temperature')}")
    print(f"   - Seed:             {client.pinned_params.get('seed')}")
    print(f"   - Reasoning Effort: {client.pinned_params.get('reasoning_effort', 'none')}")
    print(f"   - Max Tokens:       {client.pinned_params.get('max_tokens')}")

    # Check API key status
    api_key_set = bool(client.api_key)
    mode_str = "LIVE API (Particle / GLM 5.3 Flash)" if api_key_set else "OFFLINE CALIBRATED SIMULATION"
    print(f"   - Execution Mode:   {mode_str}")

    # 2. Execute Probe Query (Run 1)
    prompt = [
        {"role": "system", "content": "You are a software verification assistant. Output concise, exact code."},
        {"role": "user", "content": "Write a Python function `reverse_list(arr)` that reverses a list in-place and returns it."}
    ]

    print(f"\n2. Executing Run 1 (Seed 42)...")
    run_1 = client.generate(prompt, seed=42, arm_id="arm_0_no_memory", task_id="example_02_probe")
    print(f"   - Latency:          {run_1['latency_ms']:.2f} ms")
    print(f"   - Tokens (In/Out):  {run_1['prompt_tokens']} / {run_1['completion_tokens']}")
    print(f"   - Step Cost:        ${run_1['step_cost_usd']:.6f}")
    print(f"   - Output Snippet:\n     {run_1['response']['content'][:120]}...")

    # 3. Execute Probe Query (Run 2: Determinism Test)
    print(f"\n3. Executing Run 2 (Seed 42) to assert determinism...")
    run_2 = client.generate(prompt, seed=42, arm_id="arm_0_no_memory", task_id="example_02_probe")
    
    assert run_1["response"]["content"] == run_2["response"]["content"], "Determinism failed! Outputs differ across identical runs."
    print("   [PASS] Run 1 and Run 2 produced 100% identical outputs (Determinism Confirmed).")

    # 4. Verify Disk Telemetry Read-Back
    log_file = BASELINE_DIR / "logs" / "arm_0_no_memory.jsonl"
    print(f"\n4. Checking telemetry log on disk: {log_file.relative_to(BASELINE_DIR)}")
    assert log_file.exists() and log_file.stat().st_size > 0, "Telemetry log file was not created"
    
    with open(log_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    
    last_log = json.loads(lines[-1])
    assert last_log["response"]["content"] == run_2["response"]["content"], "Disk-persisted output does not match in-memory output"
    assert last_log["seed"] == 42
    assert last_log["temperature"] == 0.0
    print("   [PASS] Disk-persisted telemetry matches in-memory output exactly.")

    print("\n==================================================================")
    print("RESULT: Model Inference & Telemetry Pipeline Verified Accurately!")
    print("==================================================================\n")

if __name__ == "__main__":
    main()
