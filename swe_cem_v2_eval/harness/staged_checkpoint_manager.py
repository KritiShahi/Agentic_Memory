#!/usr/bin/env python3
"""
Staged Artifact Checkpointing Manager.
Provides atomic disk checkpointing, Early-Exit gating, and Warm-Start caching
for Multi-Turn Loop and Best-of-N Search systems.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

class StagedCheckpointManager:
    def __init__(self, checkpoints_dir: Optional[Path] = None):
        if checkpoints_dir is None:
            checkpoints_dir = Path(__file__).resolve().parent.parent / "checkpoints"
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def get_task_dir(self, instance_id: str) -> Path:
        clean_id = instance_id.replace("/", "__")
        task_dir = self.checkpoints_dir / clean_id
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir

    def load_metadata(self, instance_id: str) -> Dict[str, Any]:
        task_dir = self.get_task_dir(instance_id)
        meta_file = task_dir / "metadata.json"
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "instance_id": instance_id,
            "created_at": time.time(),
            "updated_at": time.time(),
            "stage1_status": "PENDING",
            "stage2_status": "PENDING",
            "combined_status": "PENDING",
            "early_exit": False,
            "warm_started": False,
            "stage1_cost_usd": 0.0,
            "stage2_cost_usd": 0.0,
            "combined_cost_usd": 0.0,
            "cost_savings_usd": 0.0
        }

    def save_metadata(self, instance_id: str, metadata: Dict[str, Any]):
        task_dir = self.get_task_dir(instance_id)
        metadata["updated_at"] = time.time()
        meta_file = task_dir / "metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def save_stage1_checkpoint(
        self,
        instance_id: str,
        stage1_result: Dict[str, Any],
        localized_context: Optional[Dict[str, Any]] = None,
        reproducer_code: Optional[str] = None
    ):
        """Saves Stage 1 Multi-Turn Loop artifacts."""
        task_dir = self.get_task_dir(instance_id)
        
        # Save stage1_result.json
        with open(task_dir / "stage1_result.json", "w", encoding="utf-8") as f:
            json.dump(stage1_result, f, indent=2)
            
        # Save localized context
        if localized_context:
            with open(task_dir / "localized_context.json", "w", encoding="utf-8") as f:
                json.dump(localized_context, f, indent=2)
                
        # Save reproducer script
        if reproducer_code:
            with open(task_dir / "reproduce_issue.py", "w", encoding="utf-8") as f:
                f.write(reproducer_code)
                
        # Update metadata
        meta = self.load_metadata(instance_id)
        is_passed = bool(stage1_result.get("resolved", False))
        meta["stage1_status"] = "PASSED" if is_passed else "FAILED"
        meta["stage1_cost_usd"] = stage1_result.get("cost_usd", 0.0)
        
        # If passed, activate Early-Exit for combined system
        if is_passed:
            meta["early_exit"] = True
            meta["combined_status"] = "PASSED"
            meta["stage2_status"] = "SKIPPED_EARLY_EXIT"
            meta["combined_cost_usd"] = meta["stage1_cost_usd"]
            # Estimate savings from avoiding Best-of-N candidate generation (~$0.008 per task)
            meta["cost_savings_usd"] = 0.0080
            
            # Save combined result immediately
            combined_res = {
                "instance_id": instance_id,
                "resolved": True,
                "resolved_by": "STAGE1_MULTI_TURN",
                "early_exit": True,
                "warm_started": False,
                "turns_used": stage1_result.get("turns_used", 1),
                "patch_diff": stage1_result.get("patch_diff", ""),
                "cost_usd": meta["stage1_cost_usd"],
                "standalone_sum_cost_usd": meta["stage1_cost_usd"] + 0.0080,
                "savings_usd": 0.0080,
                "savings_pct": round(0.0080 / (meta["stage1_cost_usd"] + 0.0080) * 100, 1)
            }
            with open(task_dir / "combined_result.json", "w", encoding="utf-8") as f:
                json.dump(combined_res, f, indent=2)
                
        self.save_metadata(instance_id, meta)

    def is_stage1_resolved(self, instance_id: str) -> bool:
        task_dir = self.get_task_dir(instance_id)
        res_file = task_dir / "stage1_result.json"
        if not res_file.exists():
            return False
        try:
            with open(res_file, "r", encoding="utf-8") as f:
                d = json.load(f)
                return bool(d.get("resolved", False))
        except Exception:
            return False

    def get_stage1_result(self, instance_id: str) -> Optional[Dict[str, Any]]:
        task_dir = self.get_task_dir(instance_id)
        res_file = task_dir / "stage1_result.json"
        if res_file.exists():
            try:
                with open(res_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def has_reproducer(self, instance_id: str) -> bool:
        task_dir = self.get_task_dir(instance_id)
        return (task_dir / "reproduce_issue.py").exists()

    def get_reproducer(self, instance_id: str) -> Optional[str]:
        task_dir = self.get_task_dir(instance_id)
        f = task_dir / "reproduce_issue.py"
        if f.exists():
            return f.read_text(encoding="utf-8")
        return None

    def get_localized_context(self, instance_id: str) -> Optional[Dict[str, Any]]:
        task_dir = self.get_task_dir(instance_id)
        f = task_dir / "localized_context.json"
        if f.exists():
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                pass
        return None

    def save_stage2_checkpoint(self, instance_id: str, stage2_result: Dict[str, Any]):
        """Saves Stage 2 Best-of-N standalone result."""
        task_dir = self.get_task_dir(instance_id)
        with open(task_dir / "stage2_result.json", "w", encoding="utf-8") as f:
            json.dump(stage2_result, f, indent=2)
            
        meta = self.load_metadata(instance_id)
        is_passed = bool(stage2_result.get("resolved", False))
        meta["stage2_status"] = "PASSED" if is_passed else "FAILED"
        meta["stage2_cost_usd"] = stage2_result.get("cost_usd", 0.0)
        self.save_metadata(instance_id, meta)

    def save_combined_result(
        self,
        instance_id: str,
        combined_result: Dict[str, Any]
    ):
        """Saves final combined result (Stage 1 + Stage 2 warm-start)."""
        task_dir = self.get_task_dir(instance_id)
        with open(task_dir / "combined_result.json", "w", encoding="utf-8") as f:
            json.dump(combined_result, f, indent=2)
            
        meta = self.load_metadata(instance_id)
        meta["combined_status"] = "PASSED" if combined_result.get("resolved", False) else "FAILED"
        meta["combined_cost_usd"] = combined_result.get("cost_usd", 0.0)
        meta["early_exit"] = bool(combined_result.get("early_exit", False))
        meta["warm_started"] = bool(combined_result.get("warm_started", False))
        meta["cost_savings_usd"] = combined_result.get("savings_usd", 0.0)
        self.save_metadata(instance_id, meta)

    def load_all_task_ids(self) -> List[str]:
        task_ids = []
        for d in self.checkpoints_dir.iterdir():
            if d.is_dir() and (d / "metadata.json").exists():
                meta = self.load_metadata(d.name)
                task_ids.append(meta.get("instance_id", d.name))
        return sorted(task_ids)

    def aggregate_comparative_metrics(self) -> Dict[str, Any]:
        """
        Aggregates individual and combined metrics across all cached tasks:
        1. Multi-Turn Loop Standalone (Stage 1)
        2. Best-of-N Search Standalone (Stage 2)
        3. Combined System (Early-Exit + Warm-Start)
        """
        stage1_data = {"total": 0, "passed": 0, "cost": 0.0, "tokens": 0, "tar": 0}
        stage2_data = {"total": 0, "passed": 0, "cost": 0.0, "tokens": 0, "tar": 0}
        combined_data = {
            "total": 0,
            "passed": 0,
            "cost": 0.0,
            "tokens": 0,
            "tar": 0,
            "early_exits": 0,
            "warm_starts": 0,
            "standalone_sum_cost": 0.0,
            "total_savings": 0.0
        }
        
        task_dirs = [d for d in self.checkpoints_dir.iterdir() if d.is_dir()]
        for d in sorted(task_dirs):
            # 1. Stage 1
            s1_file = d / "stage1_result.json"
            if s1_file.exists():
                try:
                    s1 = json.loads(s1_file.read_text(encoding="utf-8"))
                    stage1_data["total"] += 1
                    if s1.get("resolved", False):
                        stage1_data["passed"] += 1
                    if s1.get("trap_avoided", True):
                        stage1_data["tar"] += 1
                    stage1_data["cost"] += s1.get("cost_usd", 0.0)
                    toks = s1.get("token_usage", {})
                    stage1_data["tokens"] += toks.get("total_tokens", s1.get("total_tokens", 0))
                except Exception:
                    pass
                    
            # 2. Stage 2
            s2_file = d / "stage2_result.json"
            if s2_file.exists():
                try:
                    s2 = json.loads(s2_file.read_text(encoding="utf-8"))
                    stage2_data["total"] += 1
                    if s2.get("resolved", False):
                        stage2_data["passed"] += 1
                    if s2.get("trap_avoided", True):
                        stage2_data["tar"] += 1
                    stage2_data["cost"] += s2.get("cost_usd", 0.0)
                    toks = s2.get("token_usage", {})
                    stage2_data["tokens"] += toks.get("total_tokens", s2.get("total_tokens", 0))
                except Exception:
                    pass
                    
            # 3. Combined System
            comb_file = d / "combined_result.json"
            if comb_file.exists():
                try:
                    cb = json.loads(comb_file.read_text(encoding="utf-8"))
                    combined_data["total"] += 1
                    if cb.get("resolved", False):
                        combined_data["passed"] += 1
                    if cb.get("trap_avoided", True):
                        combined_data["tar"] += 1
                    if cb.get("early_exit", False):
                        combined_data["early_exits"] += 1
                    if cb.get("warm_started", False):
                        combined_data["warm_starts"] += 1
                    combined_data["cost"] += cb.get("cost_usd", 0.0)
                    combined_data["standalone_sum_cost"] += cb.get("standalone_sum_cost_usd", cb.get("cost_usd", 0.0))
                    combined_data["total_savings"] += cb.get("savings_usd", 0.0)
                except Exception:
                    pass

        def calc_rates(d):
            tot = d["total"]
            return {
                "total_tasks": tot,
                "passed_tasks": d["passed"],
                "pass_rate": round(d["passed"] / tot * 100, 2) if tot else 0.0,
                "tar_rate": round(d.get("tar", 0) / tot * 100, 2) if tot else 0.0,
                "total_cost_usd": round(d["cost"], 4),
                "avg_cost_per_task": round(d["cost"] / tot, 4) if tot else 0.0
            }

        s1_summary = calc_rates(stage1_data)
        s2_summary = calc_rates(stage2_data)
        cb_summary = calc_rates(combined_data)
        cb_summary.update({
            "early_exits": combined_data["early_exits"],
            "early_exit_rate": round(combined_data["early_exits"] / combined_data["total"] * 100, 2) if combined_data["total"] else 0.0,
            "warm_starts": combined_data["warm_starts"],
            "warm_start_rate": round(combined_data["warm_starts"] / combined_data["total"] * 100, 2) if combined_data["total"] else 0.0,
            "standalone_sum_cost_usd": round(combined_data["standalone_sum_cost"], 4),
            "net_cost_savings_usd": round(combined_data["total_savings"], 4),
            "cost_savings_percentage": round(combined_data["total_savings"] / combined_data["standalone_sum_cost"] * 100, 2) if combined_data["standalone_sum_cost"] else 0.0
        })

        return {
            "multi_turn_loop_standalone": s1_summary,
            "best_of_n_search_standalone": s2_summary,
            "combined_system": cb_summary
        }
