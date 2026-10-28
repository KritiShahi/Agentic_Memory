#!/usr/bin/env python3
"""
GLM 5.3 Flash Provider Client & Telemetry Harness.
Enforces pinned parameters (temperature=0.0, seed=42, top_p=1.0),
formats tool-calling schemas, and logs structured JSONL trajectories.
"""

import json
import os
import ssl
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

class GLM53FlashClient:
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = str(Path(__file__).resolve().parent.parent / "config" / "model_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
            
        self.model = self.config["model"]
        self.endpoint_url = self.config["endpoint_url"]
        self.pinned_params = self.config["pinned_parameters"]
        
        # Load environment variables from .env files if present
        self._load_env_files()
        self.api_key = (
            os.environ.get("PARTICLE_API_KEY")
            or os.environ.get("ZHIPU_API_KEY")
            or os.environ.get("BIGMODEL_API_KEY")
            or ""
        )
        self.logs_dir = Path(__file__).resolve().parent.parent / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Configure SSL context for HTTPS requests
        try:
            import certifi
            self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            self.ssl_context = ssl._create_unverified_context()

    def _load_env_files(self):
        """Loads environment variables from local .env files if not already set."""
        baseline_dir = Path(__file__).resolve().parent.parent
        candidate_paths = [
            baseline_dir / ".env",
            baseline_dir.parent.parent / ".env"
        ]
        for env_path in candidate_paths:
            if env_path.exists():
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k not in os.environ:
                                os.environ[k] = v

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        seed: Optional[int] = None,
        arm_id: str = "arm_0_no_memory",
        task_id: str = "task_init"
    ) -> Dict[str, Any]:
        """
        Executes a call to GLM 5.3 Flash enforcing pinned parameters and logging telemetry.
        Falls back to a deterministic offline validation response if no API key is provided.
        """
        target_seed = seed if seed is not None else self.pinned_params.get("seed", 42)
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.pinned_params["temperature"],
            "top_p": self.pinned_params["top_p"],
            "max_tokens": self.pinned_params["max_tokens"],
            "seed": target_seed
        }
        if self.pinned_params.get("reasoning_effort") and self.pinned_params["reasoning_effort"] != "none":
            payload["reasoning_effort"] = self.pinned_params["reasoning_effort"]
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        start_time = time.perf_counter()
        
        if self.api_key:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            req = urllib.request.Request(
                self.endpoint_url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            try:
                with urllib.request.urlopen(req, context=self.ssl_context, timeout=self.config.get("timeout_seconds", 60)) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
            except urllib.error.URLError as e:
                resp_data = {
                    "error": str(e),
                    "choices": [{"message": {"role": "assistant", "content": f"API Request Failed: {e}"}}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                }
        else:
            # Deterministic simulation mode for offline test calibration
            prompt_text = "".join(m.get("content", "") for m in messages)
            prompt_tokens = max(10, len(prompt_text) // 4)
            completion_tokens = 45
            resp_data = {
                "id": f"sim_glm_{int(time.time())}",
                "model": self.model,
                "seed": target_seed,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"[Simulated GLM 5.3 Flash Response | Seed={target_seed}] Execution step verified."
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        usage = resp_data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        
        pricing = self.config.get("token_pricing", {})
        input_rate = pricing.get("input_cost_per_1m_tokens", 0.10) / 1_000_000.0
        output_rate = pricing.get("output_cost_per_1m_tokens", 0.10) / 1_000_000.0
        step_cost = (prompt_tokens * input_rate) + (completion_tokens * output_rate)

        # Structured telemetry log record
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "arm_id": arm_id,
            "task_id": task_id,
            "model": self.model,
            "seed": target_seed,
            "temperature": self.pinned_params["temperature"],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_ms": round(elapsed_ms, 2),
            "step_cost_usd": round(step_cost, 6),
            "response": resp_data["choices"][0]["message"]
        }

        # Write to JSONL
        log_file = self.logs_dir / f"{arm_id}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        return log_entry

if __name__ == "__main__":
    client = GLM53FlashClient()
    test_messages = [
        {"role": "system", "content": "You are a research baseline agent."},
        {"role": "user", "content": "State the current date and verify deterministic execution."}
    ]
    res = client.generate(test_messages, arm_id="arm_0_no_memory", task_id="calibration_sanity_01")
    print("Execution Output:")
    print(json.dumps(res, indent=2))
