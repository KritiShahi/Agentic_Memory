#!/usr/bin/env python3
"""
Telemetry & Trajectory Logger for Pi Baseline Experiments.
Records turn-by-turn execution traces, token consumption, latency, and financial costs.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

class TelemetryLogger:
    def __init__(self, logs_dir: Optional[Path] = None):
        if logs_dir is None:
            self.logs_dir = Path(__file__).resolve().parent.parent / "logs"
        else:
            self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def log_turn(
        self,
        arm_id: str,
        task_id: str,
        seed: int,
        turn_index: int,
        prompt_tokens: int,
        completion_tokens: int,
        memory_injected_tokens: int,
        latency_ms: float,
        step_cost_usd: float,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_output: Optional[str] = None,
        response: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Logs a single interactive step into an arm-specific JSONL file."""
        arm_dir = self.logs_dir / arm_id
        arm_dir.mkdir(parents=True, exist_ok=True)
        
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "arm_id": arm_id,
            "task_id": task_id,
            "seed": seed,
            "turn_index": turn_index,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "memory_injected_tokens": memory_injected_tokens,
            "latency_ms": round(latency_ms, 2),
            "step_cost_usd": round(step_cost_usd, 6),
            "tool_calls": tool_calls or [],
            "tool_output": tool_output or "",
            "response": response or {}
        }
        
        task_log_file = arm_dir / f"{task_id}_seed_{seed}.jsonl"
        with open(task_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            
        return record
