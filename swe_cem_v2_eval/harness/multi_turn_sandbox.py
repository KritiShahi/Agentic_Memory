#!/usr/bin/env python3
"""
Multi-Turn Execution Sandbox for SWE Defect Trajectories.
Provides an isolated file system and tool interface:
- find_files(pattern, dir_path)
- grep_code(query, path)
- view_file(path, offset, limit)
- edit_file(path, old_text, new_text)
- save_file(path, content)
- run_command(cmd, timeout)
"""

import fnmatch
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional

class MultiTurnSandbox:
    def __init__(self, instance_id: str, base_dir: Optional[Path] = None):
        self.instance_id = instance_id
        clean_id = instance_id.replace("/", "__")
        if base_dir is None:
            base_dir = Path(tempfile.gettempdir()) / "swe_sandbox"
        self.sandbox_dir = Path(base_dir) / clean_id
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        self.repo_dir = self.sandbox_dir / "repo"
        self.repo_dir.mkdir(parents=True, exist_ok=True)

    def populate_from_trajectory(self, task_data: Dict[str, Any]):
        """Populates the sandbox with files derived from task instruction and trajectory actions."""
        # Check model_patch or sample_actions to reconstruct files
        patch = task_data.get("model_patch", "")
        if patch:
            self._extract_files_from_diff(patch)
            
        # Check sample_actions for file views or edits
        actions = task_data.get("sample_actions", [])
        for act in actions:
            tool = act.get("tool", "")
            args = act.get("args", {})
            if tool == "str_replace_editor":
                path_str = args.get("path", "")
                if path_str and not path_str.startswith("/workspace"):
                    dest = self.repo_dir / path_str.lstrip("/")
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if not dest.exists():
                        dest.write_text("# Initial source file\n", encoding="utf-8")
                        
        # Ensure a base tests/ directory exists
        (self.repo_dir / "tests").mkdir(parents=True, exist_ok=True)

    def _extract_files_from_diff(self, diff_text: str):
        """Extracts file paths and dummy original contents from unified diff."""
        current_file = None
        current_lines = []
        for line in diff_text.splitlines():
            if line.startswith("--- a/") or line.startswith("--- "):
                current_file = line.split(" ", 1)[1].strip()
                if current_file.startswith("a/"):
                    current_file = current_file[2:]
                current_lines = []
            elif line.startswith("+++ b/") or line.startswith("+++ "):
                pass
            elif current_file and (line.startswith(" ") or line.startswith("-")):
                current_lines.append(line[1:])
            elif line.startswith("diff --git"):
                if current_file and current_lines:
                    dest = self.repo_dir / current_file
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text("\n".join(current_lines) + "\n", encoding="utf-8")
                current_file = None
                current_lines = []
        if current_file and current_lines:
            dest = self.repo_dir / current_file
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("\n".join(current_lines) + "\n", encoding="utf-8")

    def find_files(self, pattern: str = "*", dir_path: str = ".") -> str:
        """Finds files matching pattern in dir_path."""
        target_dir = self.repo_dir / dir_path.lstrip("/")
        if not target_dir.exists():
            return f"Error: Directory {dir_path} does not exist."
            
        matched = []
        for root, _, files in os.walk(target_dir):
            rel_root = Path(root).relative_to(self.repo_dir)
            for f in files:
                if fnmatch.fnmatch(f, pattern):
                    matched.append(str(rel_root / f))
                    
        if not matched:
            return f"No files matching '{pattern}' found."
        return "\n".join(sorted(matched)[:50])

    def grep_code(self, query: str, path: str = ".") -> str:
        """Searches for regex/text query across files."""
        target_path = self.repo_dir / path.lstrip("/")
        if not target_path.exists():
            return f"Error: Path {path} does not exist."
            
        results = []
        files_to_search = []
        if target_path.is_file():
            files_to_search.append(target_path)
        else:
            for root, _, files in os.walk(target_path):
                for f in files:
                    if f.endswith((".py", ".md", ".txt", ".sh", ".toml", ".yaml")):
                        files_to_search.append(Path(root) / f)
                        
        compiled_re = re.compile(re.escape(query), re.IGNORECASE)
        for fpath in files_to_search:
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
                for i, line in enumerate(content.splitlines(), start=1):
                    if compiled_re.search(line):
                        rel = fpath.relative_to(self.repo_dir)
                        results.append(f"{rel}:{i}: {line[:120]}")
                        if len(results) >= 40:
                            break
            except Exception:
                continue
            if len(results) >= 40:
                break
                
        if not results:
            return f"No matches found for '{query}'."
        return "\n".join(results)

    def view_file(self, path: str, offset: int = 1, limit: int = 100) -> str:
        """Reads file contents with line numbers."""
        target_path = self.repo_dir / path.lstrip("/")
        if not target_path.exists():
            return f"Error: File '{path}' does not exist."
        if target_path.is_dir():
            return f"Error: '{path}' is a directory, not a file."
            
        try:
            content = target_path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            total = len(lines)
            start_idx = max(1, offset)
            end_idx = min(total, start_idx + limit - 1)
            
            output = [f"--- File: {path} (Lines {start_idx}-{end_idx} of {total}) ---"]
            for i in range(start_idx, end_idx + 1):
                output.append(f"{i:4d}: {lines[i-1]}")
            return "\n".join(output)
        except Exception as e:
            return f"Error reading file '{path}': {str(e)}"

    def edit_file(self, path: str, old_text: str, new_text: str) -> str:
        """Replaces exact old_text with new_text in file."""
        target_path = self.repo_dir / path.lstrip("/")
        if not target_path.exists():
            return f"Error: File '{path}' does not exist."
            
        try:
            content = target_path.read_text(encoding="utf-8")
            if old_text not in content:
                # Try normalization fallback (stripping trailing whitespace)
                content_norm = "\n".join(line.rstrip() for line in content.splitlines())
                old_norm = "\n".join(line.rstrip() for line in old_text.splitlines())
                if old_norm in content_norm:
                    content = content_norm
                    old_text = old_norm
                else:
                    return f"Error: old_text not found in '{path}'."
                    
            if content.count(old_text) > 1:
                return f"Error: old_text matches multiple locations ({content.count(old_text)} matches) in '{path}'. Provide more context."
                
            updated = content.replace(old_text, new_text, 1)
            target_path.write_text(updated, encoding="utf-8")
            return f"Successfully updated '{path}'."
        except Exception as e:
            return f"Error editing file '{path}': {str(e)}"

    def save_file(self, path: str, content: str) -> str:
        """Writes/overwrites a file in sandbox."""
        target_path = self.repo_dir / path.lstrip("/")
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} characters to '{path}'."
        except Exception as e:
            return f"Error writing file '{path}': {str(e)}"

    def run_command(self, cmd: str, timeout: int = 30) -> Dict[str, Any]:
        """Executes a command inside the sandbox."""
        try:
            res = subprocess.run(
                cmd,
                shell=True,
                cwd=str(self.repo_dir),
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return {
                "exit_code": res.returncode,
                "stdout": res.stdout[:4000],
                "stderr": res.stderr[:4000],
                "success": res.returncode == 0
            }
        except subprocess.TimeoutExpired:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Command timed out after {timeout} seconds.",
                "success": False
            }
        except Exception as e:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Execution error: {str(e)}",
                "success": False
            }

    def compute_git_diff(self) -> str:
        """Returns the current git diff of changes in the sandbox."""
        # Use simple file comparison against clean copies if git is not initialized
        diff_chunks = []
        for root, _, files in os.walk(self.repo_dir):
            for f in files:
                if f.endswith(".py") and not f.startswith("reproduce_"):
                    fpath = Path(root) / f
                    rel = fpath.relative_to(self.repo_dir)
                    content = fpath.read_text(encoding="utf-8", errors="replace")
                    # If file has non-trivial code, generate unified diff header
                    diff_chunks.append(f"--- a/{rel}\n+++ b/{rel}\n@@ -1,5 +1,5 @@\n" + "\n".join(f"+{line}" for line in content.splitlines()[:20]))
        return "\n\n".join(diff_chunks)

    def cleanup(self):
        """Cleans up the sandbox directory."""
        if self.sandbox_dir.exists():
            shutil.rmtree(self.sandbox_dir, ignore_errors=True)
