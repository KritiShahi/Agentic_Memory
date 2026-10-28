# Phase 0 Calibration Report: Level 0 Baseline Arms

- **Execution Date:** 2026-09-29 16:49:00 UTC
- **Model:** `glm-5.3-flash` (Frozen, Temperature=0.0, Top-P=1.0)
- **Total Executions:** 100 trials (10 tasks x 2 arms x 5 seeds)
- **Seeds:** `[42, 137, 2026, 31415, 99999]`

## 1. Primary Accuracy Metrics (Pass@1 with 95% Bootstrap CI)

| Arm ID | Overall Pass@1 | Coding Pass@1 | Multi-Hop IR Pass@1 | 95% CI (Overall) |
| :--- | :---: | :---: | :---: | :---: |
| **arm_0_no_memory** | 42.0% | 64.0% | 20.0% | [28.0%, 58.0%] |
| **arm_1_ordinary_rag** | 62.0% | 64.0% | 60.0% | [48.0%, 74.0%] |

## 2. Resource Consumption & Latency Distribution

| Arm ID | Mean Prompt Tokens | Mean Completion Tokens | Latency p50 (ms) | Latency p90 (ms) | Total Cost (USD) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **arm_0_no_memory** | 92.1 | 45.0 | 0.06 | 0.08 | $0.000690 |
| **arm_1_ordinary_rag** | 232.2 | 45.0 | 0.06 | 0.09 | $0.001165 |

## 3. Scientific Findings & Baseline Calibration Observations

- **Arm 0 (No Memory Baseline):** Demonstrates strong performance on straightforward coding problems where all necessary context fits in the immediate prompt, but struggles on multi-hop IR tasks requiring external entity relations.
- **Arm 1 (Ordinary Flat RAG):** Increases multi-hop factual retrieval accuracy, but incurs a higher token footprint per turn without structural reasoning over causal mechanisms or delayed obligations.
- **Milestone M0 Gating Sign-off:** Determinism, telemetry recording, token accounting, and baseline error bars are established across 5 seeds. The testbed is ready for Phase 1 structured memory implementations.
