#!/usr/bin/env python3
"""
Master Runner for All Pinned Pi Baseline Examples.
Runs:
  1. 01_verify_pinned_pi_version.py (Revision & Sandbox Verification)
  2. 02_test_single_model_query.py (Deterministic Model Probe & Telemetry)
  3. 03_solve_sample_coding_task.py (End-to-End Task & Test Harness Execution)
"""

import subprocess
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent

def run_script(script_name: str) -> bool:
    script_path = EXAMPLES_DIR / script_name
    print(f"\n>>>> RUNNING: {script_name} <<<<\n")
    proc = subprocess.run([sys.executable, str(script_path)])
    return proc.returncode == 0

def main():
    scripts = [
        "01_verify_pinned_pi_version.py",
        "02_test_single_model_query.py",
        "03_solve_sample_coding_task.py"
    ]
    
    results = {}
    for s in scripts:
        success = run_script(s)
        results[s] = "PASS" if success else "FAIL"

    print("\n==================================================================")
    print("ALL EXAMPLES EXECUTION SUMMARY")
    print("==================================================================")
    all_passed = True
    for s, status in results.items():
        print(f"{s:<36} : {status}")
        if status != "PASS":
            all_passed = False
    print("==================================================================")
    
    if all_passed:
        print("ALL EXAMPLES PASSED ACCURATELY (100% SUCCESS)\n")
        sys.exit(0)
    else:
        print("SOME EXAMPLES FAILED.\n", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
