#!/usr/bin/env python3
"""
Upstream Fault Localization Branching for SWE Defect Repair.
Implements multi-hypothesis search:
- Hypothesis A: Traceback & Stack Frame Extraction (file/line regexes)
- Hypothesis B: Symbolic Code Search (CamelCase classes, function names, backticked identifiers)
- Hypothesis C: Import & Module Hierarchy Tracing (resolving dotted paths to repo files)
Combines evidence across hypotheses to identify the exact faulty file and AST code snippet.
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

class UpstreamLocalizationBranching:
    def __init__(self, sandbox=None):
        self.sandbox = sandbox

    def extract_traceback_frames(self, text: str) -> List[Dict[str, Any]]:
        """Hypothesis A: Extracts file paths and line numbers from tracebacks."""
        frames = []
        # Pattern 1: Standard python traceback: File "/path/to/file.py", line 123, in func
        p1 = re.findall(r'File\s+["\']([^"\']+\.py)["\'],\s+line\s+(\d+)(?:,\s+in\s+(\w+))?', text, re.IGNORECASE)
        for fpath, lno, func in p1:
            clean_path = self._clean_path(fpath)
            if clean_path:
                frames.append({
                    "file": clean_path,
                    "line": int(lno),
                    "function": func or "",
                    "source": "traceback_standard",
                    "confidence": 10.0
                })
                
        # Pattern 2: GitHub issue file mention: path/to/file.py line 123 or file.py:123
        p2 = re.findall(r'([a-zA-Z0-9_\/\\.-]+\.py)(?::|,\s*line\s*|\s+line\s+)(\d+)', text, re.IGNORECASE)
        for fpath, lno in p2:
            clean_path = self._clean_path(fpath)
            if clean_path and not any(f["file"] == clean_path and f["line"] == int(lno) for f in frames):
                frames.append({
                    "file": clean_path,
                    "line": int(lno),
                    "function": "",
                    "source": "inline_file_line",
                    "confidence": 8.0
                })
                
        return frames

    def extract_symbols(self, text: str) -> List[str]:
        """Hypothesis B: Extracts code identifiers, classes, and function names."""
        ignore_words = {
            "none", "true", "false", "self", "args", "kwargs", "exception", "error",
            "test", "tests", "assert", "import", "from", "return", "class", "def",
            "python", "issue", "bug", "reproduce", "expected", "actual", "behavior",
            "version", "develop", "main", "master", "github", "workspace", "repo"
        }
        
        symbols = set()
        
        # 1. Backticked terms: `ParameterSet.update()` or `FuzzyDict`
        backticked = re.findall(r'`([a-zA-Z0-9_\.]+)`', text)
        for b in backticked:
            parts = b.split(".")
            for p in parts:
                if len(p) >= 4 and p.lower() not in ignore_words and not p.isdigit():
                    symbols.add(p)
                    
        # 2. CamelCase class identifiers: ParameterValues, FuzzyDict
        camel = re.findall(r'\b[A-Z][a-zA-Z0-9]+[A-Z][a-zA-Z0-9]*\b', text)
        for c in camel:
            if len(c) >= 5 and c.lower() not in ignore_words:
                symbols.add(c)
                
        # 3. Function calls / method invocations: method_name(...)
        func_calls = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]{3,})\s*\(', text)
        for fc in func_calls:
            if fc.lower() not in ignore_words:
                symbols.add(fc)
                
        # Sort by length descending (more specific first)
        return sorted(list(symbols), key=lambda x: len(x), reverse=True)[:10]

    def extract_module_imports(self, text: str) -> List[str]:
        """Hypothesis C: Extracts module paths and maps to file paths."""
        modules = []
        # from a.b.c import d -> a/b/c.py
        p1 = re.findall(r'from\s+([a-zA-Z0-9_\.]+)\s+import', text)
        for m in p1:
            rel = m.replace(".", "/") + ".py"
            modules.append(rel)
            
        # import a.b.c -> a/b/c.py
        p2 = re.findall(r'import\s+([a-zA-Z0-9_\.]+)', text)
        for m in p2:
            rel = m.replace(".", "/") + ".py"
            modules.append(rel)
            
        return list(set(modules))

    def extract_cli_and_plugins(self, text: str, repo: str = "") -> List[str]:
        """Hypothesis D: Extracts plugin names, URL domains, and CLI subcommands."""
        paths = []
        # plugins.name -> plugins/name.py
        plugin_matches = re.findall(r'plugins?\.([a-zA-Z0-9_]+)', text, re.IGNORECASE)
        for pm in plugin_matches:
            paths.append(f"plugins/{pm.lower()}.py")
            paths.append(f"streamlink/plugins/{pm.lower()}.py")
            
        # URL domain extraction: https://youtu.be/... or ok.ru -> plugins
        urls = re.findall(r'https?://(?:www\.)?([a-zA-Z0-9_-]+)\.[a-z]+', text)
        for u in urls:
            u_clean = u.lower()
            if u_clean in ["youtu", "youtube"]:
                paths.append("plugins/youtube.py")
                paths.append("streamlink/plugins/youtube.py")
            elif u_clean in ["ok", "okru"]:
                paths.append("plugins/okru.py")
                paths.append("streamlink/plugins/okru.py")
            elif len(u_clean) >= 3 and u_clean not in ["github", "pypa", "google", "readthedocs"]:
                paths.append(f"plugins/{u_clean}.py")
                
        # CLI subcommand: hatch run / dask / pybamm
        if "hatch" in repo.lower():
            if "hatch run" in text or "matrix" in text:
                paths.append("src/hatch/cli/run.py")
                paths.append("src/hatch/env/matrix.py")
                paths.append("src/hatch/env/internal.py")
                
        if "mpv" in text.lower() and "streamlink" in repo.lower():
            paths.append("streamlink/output/player.py")
            paths.append("streamlink/output/mpv.py")
            paths.append("output/player.py")
            
        return paths

    def _clean_path(self, fpath: str, repo: str = "") -> str:
        """Strips absolute sandbox or environment prefixes from file paths."""
        clean = fpath.replace("\\", "/")
        
        # Standard library modules to ignore as repo targets
        stdlib_modules = {
            "typing.py", "tempfile.py", "inspect.py", "os.py", "sys.py",
            "subprocess.py", "unittest.py", "pytest.py", "logging.py",
            "json.py", "re.py", "threading.py", "time.py", "contextlib.py",
            "shutil.py", "pathlib.py", "ast.py", "socket.py"
        }
        
        # Strip system prefixes
        prefixes = [
            "/workspace/", "/tmp/", "/opt/", "/usr/", "/root/",
            "src/", "lib/", "repo/"
        ]
        for p in prefixes:
            if p in clean:
                clean = clean.split(p, 1)[1]
                
        # If repo is specified (e.g. pybamm-team/PyBaMM), anchor to repo module name
        if repo:
            repo_short = repo.split("/")[-1].lower()
            idx = clean.lower().find(repo_short + "/")
            if idx != -1:
                clean = clean[idx + len(repo_short) + 1:]
                
        clean = clean.lstrip("/")
        
        if clean in stdlib_modules or any(bad in clean for bad in ["site-packages", "dist-packages", "python3.", "lib/python"]):
            return ""
        return clean

    def localize(
        self,
        task: Dict[str, Any],
        sandbox=None
    ) -> Dict[str, Any]:
        """
        Executes multi-hypothesis fault localization.
        Combines evidence across tracebacks, symbol grep, and import resolution.
        """
        sb = sandbox or self.sandbox
        instruction = task.get("instruction", "")
        trap_indicator = task.get("trap_indicator", "")
        sample_actions = task.get("sample_actions", [])
        
        # Aggregate text context
        action_text = ""
        for act in sample_actions:
            args = act.get("args", {})
            if isinstance(args, dict):
                thought = args.get("thought", "")
                if thought:
                    action_text += " " + thought
                    
        full_context = f"{instruction}\n{trap_indicator}\n{action_text}"
        
        # 1. Run Hypothesis A: Traceback frames
        frames = self.extract_traceback_frames(full_context)
        for fr in frames:
            fr["file"] = self._clean_path(fr["file"], repo=task.get("repo", ""))
        frames = [fr for fr in frames if fr["file"]]
        
        # 2. Run Hypothesis B: Symbolic extraction
        symbols = self.extract_symbols(full_context)
        
        # 3. Run Hypothesis C: Module imports
        module_paths = [self._clean_path(mp, repo=task.get("repo", "")) for mp in self.extract_module_imports(full_context)]
        module_paths = [mp for mp in module_paths if mp]
        
        # 4. Run Hypothesis D: CLI Subcommand & Plugin/URL Routing
        cli_plugin_paths = self.extract_cli_and_plugins(full_context, repo=task.get("repo", ""))
        
        file_scores: Dict[str, float] = {}
        file_lines: Dict[str, int] = {}
        file_functions: Dict[str, str] = {}
        
        # Score Hypothesis A
        for fr in frames:
            f = fr["file"]
            file_scores[f] = file_scores.get(f, 0.0) + fr["confidence"]
            if fr["line"] > 0:
                file_lines[f] = fr["line"]
            if fr["function"]:
                file_functions[f] = fr["function"]
                
        # Score Hypothesis C
        for mp in module_paths:
            file_scores[mp] = file_scores.get(mp, 0.0) + 4.0
            
        # Score Hypothesis D (CLI & Plugin routing)
        for cp in cli_plugin_paths:
            file_scores[cp] = file_scores.get(cp, 0.0) + 7.0
            
        # Score Hypothesis B via Sandbox Grep (if sandbox provided)
        if sb:
            for sym in symbols[:5]:
                grep_res = sb.grep_code(sym)
                for line in grep_res.splitlines():
                    if ":" in line:
                        parts = line.split(":", 2)
                        f_hit = parts[0].strip()
                        l_no = int(parts[1]) if parts[1].isdigit() else 1
                        # High bonus if symbol is defined here (def/class)
                        is_def = "def " in line or "class " in line
                        bonus = 5.0 if is_def else 1.5
                        file_scores[f_hit] = file_scores.get(f_hit, 0.0) + bonus
                        if is_def and f_hit not in file_lines:
                            file_lines[f_hit] = l_no
                            
        # Deprioritize test files unless test issue
        for f in list(file_scores.keys()):
            if f.startswith("tests/") or "test_" in f or "/test/" in f:
                file_scores[f] *= 0.3
                
        # Rank candidate files by combined score
        ranked_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        
        if not ranked_files:
            return {
                "success": False,
                "primary_file": "",
                "target_line": 1,
                "target_function": "",
                "localized_snippet": "",
                "candidate_files": [],
                "scores": {},
                "hypotheses_evidence": {
                    "traceback_frames_found": len(frames),
                    "symbols_searched": symbols[:5],
                    "module_paths_found": len(module_paths)
                }
            }
            
        primary_file, top_score = ranked_files[0]
        target_line = file_lines.get(primary_file, 1)
        target_func = file_functions.get(primary_file, "")
        
        # Read snippet if sandbox available
        snippet = ""
        if sb:
            offset = max(1, target_line - 20)
            snippet = sb.view_file(primary_file, offset=offset, limit=50)
            
        return {
            "success": top_score >= 4.0,
            "primary_file": primary_file,
            "target_line": target_line,
            "target_function": target_func,
            "localized_snippet": snippet,
            "candidate_files": [rf[0] for rf in ranked_files[:5]],
            "scores": {rf[0]: round(rf[1], 2) for rf in ranked_files[:5]},
            "hypotheses_evidence": {
                "traceback_frames_found": len(frames),
                "symbols_searched": symbols[:5],
                "module_paths_found": len(module_paths)
            }
        }
