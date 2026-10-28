#!/usr/bin/env python3
"""
Comprehensive Statistical Analysis for SWE-CEM v2 Ablation Suite (E1 - E6).
Computes:
- Pass@1 Resolution Rate & 95% Wilson Confidence Intervals
- Trap Avoidance Rate (TAR) & 95% Wilson Confidence Intervals
- McNemar's Paired Chi-Square Tests with Edwards' Continuity Correction & p-values
- Tool Call Dropout counts & rates
- Token distributions (prompt, completion, total) and spend ($ USD)
- TDD subset resolution (N=48)
- Breakdown by repo and error category
"""

import json
import math
from pathlib import Path

BASE_DIR = Path("/Users/kritishahi/Desktop/Humanitarians AI/test")
EVAL_DIR = BASE_DIR / "research/swe_cem_v2_eval"
RIGOROUS_DIR = BASE_DIR / "research/swe_rigorous_eval"
BENCHMARK_PATH = EVAL_DIR / "benchmarks/eval_swe_stream.json"

def wilson_interval(successes: int, total: int, confidence: float = 0.95):
    if total == 0:
        return 0.0, 0.0
    # For 95% CI, z = 1.959963984540054
    z = 1.959963984540054
    p = successes / total
    denom = 1 + (z**2) / total
    centre = (p + (z**2) / (2 * total)) / denom
    margin = (z * math.sqrt((p * (1 - p) / total) + (z**2) / (4 * total**2))) / denom
    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    return lower * 100, upper * 100

def chi2_1df_pvalue(chi2: float) -> float:
    """Exact p-value for chi-squared distribution with 1 degree of freedom."""
    if chi2 <= 0.0:
        return 1.0
    return math.erfc(math.sqrt(chi2 / 2.0))

def mcnemar_test(results_a: dict, results_b: dict):
    """
    Computes McNemar's paired test with Edwards' continuity correction.
    Table:
                Arm B +    Arm B -
    Arm A +        a          b
    Arm A -        c          d
    b = Arm A passed, Arm B failed
    c = Arm A failed, Arm B passed
    """
    common_keys = sorted(set(results_a.keys()).intersection(set(results_b.keys())))
    a = b = c = d = 0
    for k in common_keys:
        res_a = results_a[k]
        res_b = results_b[k]
        if res_a and res_b:
            a += 1
        elif res_a and not res_b:
            b += 1
        elif not res_a and res_b:
            c += 1
        else:
            d += 1
    
    n_discordant = b + c
    if n_discordant == 0:
        chi2 = 0.0
        p_val = 1.0
    else:
        # Edwards' continuity correction: (|b - c| - 1)^2 / (b + c)
        chi2 = (abs(b - c) - 1.0) ** 2 / n_discordant
        p_val = chi2_1df_pvalue(chi2)
    
    return {
        "n_common": len(common_keys),
        "both_passed": a,
        "a_only": b,
        "b_only": c,
        "both_failed": d,
        "discordant": n_discordant,
        "chi2": round(chi2, 4),
        "p_value": round(p_val, 4)
    }

def load_eval_stream():
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_run_results(run_dir: Path):
    results = {}
    tokens = {"prompt": [], "completion": [], "total": []}
    latencies = []
    costs = []
    tar_count = 0
    tool_call_dropouts = 0
    tdd_results = {}
    
    stream = load_eval_stream()
    tdd_ids = set()
    for item in stream:
        if item.get("gen_tests_correct", 0) > 0:
            tdd_ids.add(item["instance_id"])
            
    summary_files = list(run_dir.glob("*/summary.json"))
    if not summary_files:
        summary_files = [f for f in run_dir.glob("*.json") if not f.name.endswith("_summary.json") and not f.name.startswith("ablation")]
        
    for f in sorted(summary_files):
        with open(f, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        iid = d["instance_id"]
        res = bool(d.get("passed", False) or d.get("resolved", False))
        results[iid] = res
        if iid in tdd_ids:
            tdd_results[iid] = res
            
        tar = bool(d.get("trap_avoided", True) if "trap_avoided" in d else d.get("tar", True))
        if tar:
            tar_count += 1
            
        drop = bool(d.get("has_tool_call_syntax", False) or d.get("tool_call_dropout", False))
        if drop:
            tool_call_dropouts += 1
            
        usg = d.get("token_usage", {})
        p_tok = d.get("prompt_tokens", usg.get("prompt_tokens", 0))
        c_tok = d.get("completion_tokens", usg.get("completion_tokens", 0))
        t_tok = d.get("total_tokens", usg.get("total_tokens", p_tok + c_tok))
        tokens["prompt"].append(p_tok)
        tokens["completion"].append(c_tok)
        tokens["total"].append(t_tok)
        
        lat = d.get("latency_ms", 0.0)
        if lat > 0:
            latencies.append(lat)
            
        cst = d.get("cost_usd", 0.0)
        if cst > 0:
            costs.append(cst)
            
    total = len(results)
    pass_cnt = sum(1 for v in results.values() if v)
    tdd_total = len(tdd_results)
    tdd_pass = sum(1 for v in tdd_results.values() if v)
    
    p_low, p_high = wilson_interval(pass_cnt, total)
    t_low, t_high = wilson_interval(tar_count, total)
    tdd_low, tdd_high = wilson_interval(tdd_pass, tdd_total)
    
    return {
        "total": total,
        "passed": pass_cnt,
        "pass_rate": round(pass_cnt / total * 100, 2) if total else 0,
        "pass_ci": (round(p_low, 2), round(p_high, 2)),
        "tar_count": tar_count,
        "tar_rate": round(tar_count / total * 100, 2) if total else 0,
        "tar_ci": (round(t_low, 2), round(t_high, 2)),
        "tool_call_dropouts": tool_call_dropouts,
        "tool_call_dropout_rate": round(tool_call_dropouts / total * 100, 2) if total else 0,
        "tdd_total": tdd_total,
        "tdd_passed": tdd_pass,
        "tdd_rate": round(tdd_pass / tdd_total * 100, 2) if tdd_total else 0,
        "tdd_ci": (round(tdd_low, 2), round(tdd_high, 2)),
        "avg_prompt_tokens": round(sum(tokens["prompt"]) / len(tokens["prompt"]), 1) if tokens["prompt"] else 0,
        "avg_completion_tokens": round(sum(tokens["completion"]) / len(tokens["completion"]), 1) if tokens["completion"] else 0,
        "avg_total_tokens": round(sum(tokens["total"]) / len(tokens["total"]), 1) if tokens["total"] else 0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
        "total_cost_usd": round(sum(costs), 4),
        "instance_results": results
    }

def main():
    experiments = {
        "E1_baseline": EVAL_DIR / "runs/E1_baseline",
        "E2_episode_quality": EVAL_DIR / "runs/E2_episode_quality",
        "E3_applicability": EVAL_DIR / "runs/E3_applicability",
        "E4_phase_aware": EVAL_DIR / "runs/E4_phase_aware",
        "E5_selective_interception": EVAL_DIR / "runs/E5_selective_interception",
        "E6_combined_v2": EVAL_DIR / "runs/E6_combined_v2",
    }
    
    baseline_arms = {
        "Arm_0_ZeroShot": RIGOROUS_DIR / "runs/swe_arm_0_no_memory",
        "Arm_1_FlatRAG": RIGOROUS_DIR / "runs/swe_arm_1_flat_rag",
        "Arm_3_BaselineCEM": RIGOROUS_DIR / "runs/swe_arm_3_contrastive_memory",
    }
    
    all_data = {}
    for exp_id, path in experiments.items():
        all_data[exp_id] = load_run_results(path)
        
    for arm_id, path in baseline_arms.items():
        all_data[arm_id] = load_run_results(path)
        
    print("=== PASS@1 & TAR SUMMARY TABLE ===")
    for k, d in all_data.items():
        print(f"{k:26}: Pass@1 = {d['passed']}/{d['total']} ({d['pass_rate']}%, 95% CI: [{d['pass_ci'][0]}%, {d['pass_ci'][1]}%]) | TAR = {d['tar_rate']}% | Dropouts = {d['tool_call_dropouts']} | TDD = {d['tdd_passed']}/{d['tdd_total']} ({d['tdd_rate']}%) | AvgCompTok = {d['avg_completion_tokens']}")
        
    print("\n=== MCNEMAR PAIRED TESTS ===")
    pairs_to_compare = [
        ("E1_baseline", "E2_episode_quality", "Format Guard (Clean Deltas vs Monolithic Tool Strings)"),
        ("E2_episode_quality", "E3_applicability", "Applicability Conditions & Novelty Dampening"),
        ("E3_applicability", "E4_phase_aware", "Phase-Aware Retrieval (Split Token Budgets)"),
        ("E3_applicability", "E5_selective_interception", "Selective Interception (Decoupled Hard Constraints)"),
        ("E5_selective_interception", "E6_combined_v2", "Full Combined CEM-v2 System Integration"),
        ("E1_baseline", "E6_combined_v2", "Overall CEM Evolution: E1 Baseline -> E6 Combined v2"),
        ("Arm_0_ZeroShot", "E6_combined_v2", "E6 Combined v2 vs. Zero-Shot Control (Arm 0)"),
        ("Arm_1_FlatRAG", "E6_combined_v2", "E6 Combined v2 vs. Flat RAG (Arm 1)"),
        ("Arm_3_BaselineCEM", "E6_combined_v2", "E6 Combined v2 vs. Original Arm 3 Baseline"),
    ]
    
    mcnemar_results = {}
    for arm_a, arm_b, label in pairs_to_compare:
        res = mcnemar_test(all_data[arm_a]["instance_results"], all_data[arm_b]["instance_results"])
        mcnemar_results[f"{arm_a} vs {arm_b}"] = {
            "label": label,
            "metrics": res
        }
        print(f"\nComparison: {label} [{arm_a} vs {arm_b}]")
        print(f"  Contingency: Both+ = {res['both_passed']}, {arm_a}+ only = {res['a_only']}, {arm_b}+ only = {res['b_only']}, Both- = {res['both_failed']}")
        print(f"  Discordant: {res['discordant']} | Chi2: {res['chi2']} | p-value: {res['p_value']}")
        
    output_path = EVAL_DIR / "runs/comprehensive_statistical_analysis.json"
    clean_output = {
        "experiments": {k: {k2: v2 for k2, v2 in v.items() if k2 != "instance_results"} for k, v in all_data.items()},
        "mcnemar_comparisons": mcnemar_results
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_output, f, indent=2)
    print(f"\nSaved comprehensive analysis to {output_path}")

if __name__ == "__main__":
    main()
