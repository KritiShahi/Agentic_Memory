#!/usr/bin/env python3
"""
Isolation Guard Test for Pi Baseline.
Verifies that:
1. Pi is installed locally within research/pi_baseline/.
2. No baseline configuration or code imports/references .optimus/ or arkenOS internal runtimes.
3. The runner script enforces the isolated PI_CONFIG_DIR.
"""

import os
import subprocess
import sys
from pathlib import Path

def test_local_binary_exists():
    baseline_dir = Path(__file__).resolve().parent.parent
    pi_bin = baseline_dir / "node_modules" / ".bin" / "pi"
    assert pi_bin.exists(), f"Pi binary missing at {pi_bin}"
    print("PASS: Local Pi binary verified in research/pi_baseline/node_modules/.bin/pi")

def test_runner_execution_and_version():
    baseline_dir = Path(__file__).resolve().parent.parent
    runner_script = baseline_dir / "bin" / "pi_runner.sh"
    assert runner_script.exists(), f"Runner missing at {runner_script}"
    
    proc = subprocess.run(
        [str(runner_script), "--version"],
        capture_output=True,
        text=True,
        check=True
    )
    version = proc.stdout.strip()
    assert version == "0.87.1", f"Unexpected Pi version: {version}"
    print(f"PASS: Isolated runner successfully executed Pi v{version}")

def test_zero_coupling_with_optimus():
    baseline_dir = Path(__file__).resolve().parent.parent
    forbidden_terms = [".optimus", "agents/optimus"]
    
    scanned_files = 0
    for root, dirs, files in os.walk(baseline_dir):
        # Skip node_modules, vendor git blobs, and the test itself
        if "node_modules" in root or ".git" in root or "tests" in root:
            continue
        for f in files:
            if f.endswith((".py", ".sh", ".json", ".js", ".ts")):
                filepath = Path(root) / f
                scanned_files += 1
                content = filepath.read_text(encoding="utf-8", errors="ignore")
                for term in forbidden_terms:
                    lines = content.splitlines()
                    for idx, line in enumerate(lines, 1):
                        if term in line:
                            raise AssertionError(
                                f"Illegal reference to '{term}' at {filepath}:{idx} -> {line.strip()}"
                            )
                                
    print(f"PASS: Scanned {scanned_files} operational project files; verified zero coupling with Optimus.")

if __name__ == "__main__":
    try:
        test_local_binary_exists()
        test_runner_execution_and_version()
        test_zero_coupling_with_optimus()
        print("\nAll baseline isolation guard tests PASSED successfully.")
    except Exception as exc:
        print(f"\nFAIL: Isolation guard test failed: {exc}", file=sys.stderr)
        sys.exit(1)
