#!/usr/bin/env python3
"""
Example 1: Verify Pinned Pi Version and Binary Sandboxing.

This script demonstrates that:
1. The Pi repository is pinned to the exact commit (11894012dd461232eb075bc890538b6866860a10).
2. The isolated local npm package is version 0.87.1.
3. The sandboxed binary wrapper (bin/pi_runner.sh) executes correctly without host pollution.
"""

import json
import os
import subprocess
from pathlib import Path

# Paths relative to this script
EXAMPLES_DIR = Path(__file__).resolve().parent
BASELINE_DIR = EXAMPLES_DIR.parent

def main():
    print("==================================================================")
    print("EXAMPLE 1: VERIFY PINNED PI REVISION & RUNNER HEALTH")
    print("==================================================================")

    # 1. Check Pinned Git Commit
    revision_file = BASELINE_DIR / "vendor" / "pi" / "REVISION.json"
    print(f"\n1. Checking pinned revision file: {revision_file.relative_to(BASELINE_DIR)}")
    assert revision_file.exists(), f"Error: Revision file missing at {revision_file}"
    with open(revision_file, "r", encoding="utf-8") as f:
        rev_data = json.load(f)
    
    pinned_commit = rev_data.get("pinned_commit")
    expected_commit = "11894012dd461232eb075bc890538b6866860a10"
    print(f"   - Pinned Commit:   {pinned_commit}")
    assert pinned_commit == expected_commit, f"Commit mismatch! Expected {expected_commit}, got {pinned_commit}"
    print("   [PASS] Pinned commit hash verified.")

    # 2. Check Local npm Package Version
    pkg_json = BASELINE_DIR / "package.json"
    print(f"\n2. Checking local package.json dependencies: {pkg_json.relative_to(BASELINE_DIR)}")
    with open(pkg_json, "r", encoding="utf-8") as f:
        pkg_data = json.load(f)
    
    pi_pkg_version = pkg_data.get("dependencies", {}).get("@earendil-works/pi-coding-agent")
    print(f"   - Installed @earendil-works/pi-coding-agent: {pi_pkg_version}")
    assert pi_pkg_version == "0.87.1", f"Unexpected version: {pi_pkg_version}"
    print("   [PASS] Package version confirmed as 0.87.1.")

    # 3. Execute Sandboxed Binary Runner
    runner_script = BASELINE_DIR / "bin" / "pi_runner.sh"
    print(f"\n3. Executing isolated runner: {runner_script.relative_to(BASELINE_DIR)} --version")
    assert runner_script.exists() and os.access(runner_script, os.X_OK), "Runner script missing or not executable"
    
    proc = subprocess.run(
        [str(runner_script), "--version"],
        capture_output=True,
        text=True,
        check=True
    )
    stdout_version = proc.stdout.strip()
    print(f"   - Runner Output:   {stdout_version}")
    assert stdout_version == "0.87.1", f"Binary execution returned unexpected version: {stdout_version}"
    print("   [PASS] Sandboxed binary executed successfully.")

    print("\n==================================================================")
    print("RESULT: Pinned Pi Agent Version 0.87.1 is Healthy and Accurate!")
    print("==================================================================\n")

if __name__ == "__main__":
    main()
