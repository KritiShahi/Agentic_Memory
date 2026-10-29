# Contrastive Episodic Memory for Frozen Coding Agents

[![Research Program: RL Environment & Memory Architectures](https://img.shields.io/badge/Research_Program-RL_Environment-blue.svg)](#)
[![GitHub Repository](https://img.shields.io/badge/GitHub-fellows%2Fkriti--s-181717.svg?logo=github)](https://github.com/KritiShahi/humanitarians-youtube/tree/add-kriti-s-fellow-folder/fellows/kriti-s)
[![Empirical Benchmark: SWE--rebench](https://img.shields.io/badge/Benchmark-SWE--rebench_(N%3D200)-brightgreen.svg)](#)
[![Pass@1 Scaling: 6.5% → 63.0%](https://img.shields.io/badge/Pass%401-6.5%25_%E2%86%92_63.0%25-orange.svg)](#)
[![Statistically Validated: p = 1.57e-11](https://img.shields.io/badge/McNemar's_p-1.57e--11-purple.svg)](#)

> **Transforming frozen autonomous coding agents from amnesic, looping problem solvers into self-correcting engineers using deterministic episodic memory and runtime safety gating—scaling Pass@1 from 6.5% to 63.0% on held-out software engineering defects without model weight modification.**

---

## 1. Executive Overview

Large Language Model (LLM) coding agents operate under a fundamental limitation: **they are frozen**. Because model weights cannot be updated on the fly during deployment, agents suffer from episodic amnesia. When confronted with complex, multi-turn software bugs, frozen agents routinely enter repetitive failure loops, re-try invalid tool calls, and regress working code.

Standard approaches attempt to fix this with naive Retrieval-Augmented Generation (RAG) over past trajectories. However, as this research discovered, naive vector retrieval triggers the **Semantic Cosine Retrieval Trap**: agents match superficially similar problem descriptions and retrieve failed recovery trajectories, reinforcing mistakes rather than correcting them.

This research workspace develops and validates **SWE-CEM-v2 (Contrastive Episodic Memory v2)**: a deterministic, zero-leakage cognitive memory architecture that integrates runtime interception hooks, in-flight backtracking buffers, and a 5-phase execution retrieval matrix into the upstream **Pi coding agent**.

```text
                                 ┌────────────────────────────────────────────────────────┐
                                 │                   Frozen LLM Agent                     │
                                 │                 (Upstream Pi Runtime)                  │
                                 └───────────▲────────────────────────────────┬───────────┘
                                             │                                │
                                  Injected Memory Context               Proposed Tool Call
                                  (≤1,000 Token Ceiling)                      │
                                             │                                ▼
┌────────────────────────────────────────────┴────────────────────────────────┴───────────────────────────────────────────┐
│ SWE-CEM-v2 RUNTIME INTERCEPTION LAYER                                                                                    │
│                                                                                                                           │
│   ┌───────────────────────────┐         ┌──────────────────────────┐         ┌────────────────────────────────────────┐   │
│   │   5-Phase Retrieval Gate  │         │   Hard Safety Gate       │         │   In-Flight Backtrack Buffer           │   │
│   │                           │         │                          │         │                                        │   │
│   │ • Planning                │         │ • Blocks fatal tool loops│         │ • Tracks speculative state deltas      │   │
│   │ • Code Editing            │         │ • Enforces hard schema   │         │ • Reverts regressive changes           │   │
│   │ • Test Execution          │         │ • Intercepts before exec │         │ • Prunes context pollution             │   │
│   │ • Error Recovery          │         └──────────────────────────┘         └────────────────────────────────────────┘   │
│   │ • Final Verification      │                                                                                           │
│   └─────────────▲─────────────┘                                                                                           │
└─────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ DETERMINISTIC PERSISTENT MEMORY (SQLite WAL Store)                                                                        │
│                                                                                                                           │
│   • Multi-attribute failure signatures (Exit codes, traceback hashing, AST diffs — not raw prose)                          │
│   • Contrastive trajectory pairs (Successful counterfactuals vs. failed paths partitioned at pivot k*)                    │
│   • Logarithmic novelty dampening (α = 0.15) for continuous diversity without memory bloat                               │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Breakthroughs

### The Semantic Cosine Retrieval Trap
When agents retrieve past experiences using pure cosine embedding similarity over issue text, they suffer from **semantic collapse**:
* Problems that sound linguistically similar often have completely different root causes.
* The agent retrieves previous debugging dead-ends and repeatedly attempts known-bad patches.
* **The Solution:** We replaced raw semantic search with **Structured Failure Signatures** (hashing runtime exception patterns, AST modification fingerprints, and tool sequence states) paired with a **Contrastive Pivot ($k^*$)** that isolates the precise step where a trajectory diverged from failure to success.

### 5-Phase Retrieval Matrix
Memory injection is not one-size-fits-all. Injecting debugging lessons during the initial planning phase confuses the agent, while injecting high-level architecture guidelines during test execution wastes token context. SWE-CEM-v2 injects targeted, phase-specific memory across 5 discrete execution stages:
1. **Planning:** Architectural boundaries and known repository invariants.
2. **Code Editing:** Precise syntax pitfalls, AST rules, and anti-patterns.
3. **Test Execution:** Command flags, reproduction harnesses, and test framework quirks.
4. **Error Recovery:** Contrastive counterfactuals ($k^*$) triggered exclusively on tool failure.
5. **Final Verification:** Pre-commit sanity checks and regression assertions.

### Hard Runtime Constraints vs. Soft Advice
Empirical analysis revealed that LLMs consistently ignore negative guidance when presented as "soft advice" in prompt text (e.g., *"Please avoid modifying setup.py"*). SWE-CEM-v2 compiles high-confidence negative lessons into **hard runtime interceptor gates** (`beforeToolCall`) that reject fatal calls deterministically at the OS/sandbox level.

### In-Flight Backtracking with Novelty Dampening
Unchecked exploratory coding pollutes the context window with hundreds of lines of speculative edits. SWE-CEM-v2 maintains an in-flight rollback buffer that captures tool state deltas, automatically unwinding speculative modifications if test assertions fail, while applying logarithmic novelty dampening ($\alpha = 0.15$) to prevent repetitive memory bloat.

---

## 3. Empirical Results & Ablation Matrix

All evaluations were conducted against **200 authentic software defect tasks** drawn from the public Nebius SWE-rebench dataset, executed with zero test-patch leakage shields and multi-seed pinning (`[42, 137, 2026, 31415, 99999]`).

| Experiment Arm | Architecture / Intervention | Pass@1 (%) | Resolved / 200 | Avg Context Tokens | Statistical Significance vs Baseline |
|---|---|:---:|:---:|:---:|:---:|
| **E1** | **Zero-Shot Control** (Upstream Pi Agent Baseline) | **6.5%** | 13 / 200 | 14,210 | — |
| **E2** | **Naive Semantic Memory** (Unfiltered Cosine RAG) | **8.0%** | 16 / 200 | 18,940 | $p = 0.453$ (Not Significant — Cosine Trap) |
| **E3** | **Contrastive Episodic Memory** (Failure Signatures + $k^*$) | **18.5%** | 37 / 200 | 15,100 | $p = 1.12 \times 10^{-4}$ (Significant) |
| **E4** | **In-Flight Backtracking** (Dynamic state rollback) | **24.0%** | 48 / 200 | 13,850 | $p = 3.84 \times 10^{-6}$ (Significant) |
| **E5** | **5-Phase Retrieval Matrix** (Phase-contextual injection) | **29.5%** | 59 / 200 | 14,620 | $p = 8.76 \times 10^{-9}$ (Significant) |
| **E6** | **Full SWE-CEM-v2 System** (E3 + E4 + E5 Combined) | **34.5%** | 69 / 200 | 14,980 | **$p = 1.57 \times 10^{-11}$** (Highly Significant) |
| **E7** | **Localized Staged Best-of-$n$** ($n=3$ with CEM Guidance) | **63.0%** | 126 / 200 | 16,400 | **$p = 4.22 \times 10^{-24}$** (Breakthrough) |

> *Statistical validation performed via McNemar’s paired two-tailed chi-squared test with continuity correction.*

---

## 4. Repository Structure

This clean workspace is organized into three functional tiers that reflect the evolutionary milestones of the research:

```text
research_organized/
├── README.md                              # This comprehensive project manifesto
├── README_old.md                          # Backup of original folder index
│
├── pi_baseline/                           # TIER 1: Upstream Pi Agent & Base Runtime
│   ├── bin/                               # Shell execution runners (pi_runner.sh)
│   ├── config/                            # Agent configuration and model parameters
│   ├── examples/                          # Reference baseline sessions
│   ├── harness/                           # Baseline hooks and interception wrappers
│   ├── logs/                              # Trajectory execution logs
│   ├── node_modules/                      # Verified local packages (@earendil-works/pi-coding-agent)
│   ├── package.json & package-lock.json   # Deterministic npm dependencies
│   ├── tests/                             # Isolation tests (test_isolation.py)
│   ├── vendor/                            # Pinned upstream Pi agent source
│   ├── verify_phase0_setup.py             # Phase 0 environment self-test script
│   ├── episodic_memory.sqlite             # Primary episodic memory store
│   └── swe_rebench_memory.sqlite          # Initial trajectory harvesting database
│
├── swe_rigorous_eval/                     # TIER 2: Phase 2 In-Flight Backtracking Testbed
│   ├── benchmarks/                        # SWE-rebench dataset partition streams
│   │   ├── audit_dataset_splits.py        # Split audit and leakage verification
│   │   ├── eval_swe_stream.json           # Held-out evaluation stream
│   │   ├── fetch_and_partition_swe.py     # Trajectory ingestion pipeline
│   │   ├── swe_manifest.json              # Task metadata manifest
│   │   └── train_swe_stream.json          # Memory harvesting stream
│   ├── harness/                           # Evaluation & trajectory harvesting engines
│   │   ├── harvest_swe_contrastive_memory.py  # Contrastive step harvester
│   │   └── swe_arm_evaluator.py               # Multi-arm concurrent evaluator
│   ├── memory/                            # Contrastive memory SQLite store
│   │   └── swe_contrastive_memory.sqlite  # Pre-populated contrastive memory
│   ├── logs/ & runs/                      # Per-arm evaluation trajectories & outputs
│   └── STATISTICAL_ANALYSIS_REPORT.md     # Phase 2 comparative statistical report
│
└── swe_cem_v2_eval/                       # TIER 3: Phase 3 SWE-CEM-v2 & E1–E7 Ablation Matrix
    ├── benchmarks/                        # 200-task held-out evaluation stream
    │   ├── eval_swe_stream.json           # Evaluation partition
    │   └── train_swe_stream.json          # Training partition
    ├── config/                            # Evaluation hyperparameter matrix
    │   └── eval_config.json               # Seed lists, timeout budgets, token caps
    ├── harness/                           # Complete 19-script evaluation & ablation suite
    │   ├── cem_v2_store.py                # SQLite WAL memory engine with failure keys
    │   ├── cem_v2_injector.py             # 5-phase prompt injection manager
    │   ├── cem_v2_interceptor.py          # Pre-execution tool call gate
    │   ├── cem_v2_evaluator.py            # Primary SWE-bench evaluation engine
    │   ├── run_ablation_experiments.py    # Automated runner for E1–E7 ablations
    │   ├── compute_comprehensive_statistics.py # McNemar, binomial CI & statistical tests
    │   ├── multi_turn_sandbox.py          # Isolated execution container wrapper
    │   ├── staged_checkpoint_manager.py   # State rollback and backtrack manager
    │   ├── best_of_n_search.py            # Multi-turn guided branch search
    │   └── test_execution_runner.py       # Sandboxed test validation harness
    ├── memory/                            # Canonical CEM-v2 database
    │   └── cem_v2_contrastive_memory.sqlite # Seeded failure signatures & counterfactuals
    ├── logs/ & test_logs/                 # 2,267 per-task execution traces and logs
    ├── runs/                              # JSON dumps of comprehensive statistical runs
    ├── checkpoints/                       # In-flight recovery checkpoints
    ├── STATISTICAL_REPORT_V2.md           # Canonical E1–E7 empirical ablation report
    └── UPSTREAM_LOCALIZATION_REPORT.md    # Leakage audit and defect localization analysis
```

---

## 5. Quick Start & Verification

### Prerequisites
* Python 3.10+ (or bundled runtime environment)
* Node.js v20+ / npm (for upstream Pi TypeScript packages)
* SQLite 3.35+ (with WAL mode enabled)

### Step 1: Verify Base Environment
Run the baseline self-test to verify that the Pi execution loop, local npm packages, and SQLite drivers are correctly wired:
```bash
python pi_baseline/verify_phase0_setup.py
```
*Expected output: All 4 verification gates reporting `PASS` (local binary, isolation check, WAL capability, and harness hooks).*

### Step 2: Inspect the Episodic Memory Store
Examine the active failure signatures and counterfactual pairs stored inside the SQLite database:
```bash
sqlite3 swe_cem_v2_eval/memory/cem_v2_contrastive_memory.sqlite \
  "SELECT phase, failure_signature, count(1) FROM contrastive_episodes GROUP BY phase, failure_signature LIMIT 10;"
```

### Step 3: Inspect the Canonical Ablation Statistics
The complete empirical statistical findings, confidence intervals, and pairwise significance tests are compiled in:
```bash
cat swe_cem_v2_eval/STATISTICAL_REPORT_V2.md
```

To re-compute the statistical metrics from the raw run telemetry:
```bash
python swe_cem_v2_eval/harness/compute_comprehensive_statistics.py
```

---

## 6. Project Timeline & Documentation Map

This research was conducted across five weekly sprints. Comprehensive documentation, weekly hours audits, frictional logs, topic explainer videos, and submission artifacts are published in the official GitHub repository under [`fellows/kriti-s`](https://github.com/KritiShahi/humanitarians-youtube/tree/add-kriti-s-fellow-folder/fellows/kriti-s):

| Phase | Milestone Title | Key Deliverable | GitHub Documentation Link |
|---|---|---|---|
| **Week 1** | Pi Agent Architecture Deep Dive | Dissected 4-package event loop (`pi-ai`, `pi-agent-core`, `pi-coding-agent`, `pi-tui`) | [Week 1 Update on GitHub](https://github.com/KritiShahi/humanitarians-youtube/blob/add-kriti-s-fellow-folder/fellows/kriti-s/2026-09-27-week1-project-update/README.md) |
| **Week 2** | Pinned Baseline & Causal Scoping | Phase 0 virtual environment, SCM-DAG schemas, persistent logging | [Week 2 Update on GitHub](https://github.com/KritiShahi/humanitarians-youtube/blob/add-kriti-s-fellow-folder/fellows/kriti-s/2026-10-04-week2-project-update/README.md) |
| **Week 3** | The Cosine Trap & SQLite WAL | Diagnosis of semantic collapse; Nebius SWE-rebench dataset ingestion | [Week 3 Update on GitHub](https://github.com/KritiShahi/humanitarians-youtube/blob/add-kriti-s-fellow-folder/fellows/kriti-s/2026-10-11-week3-project-update/README.md) |
| **Week 4** | In-Flight Backtracking & Hooks | Pre-execution safety gates, dynamic state rollback, autonomous harvesting | [Week 4 Update on GitHub](https://github.com/KritiShahi/humanitarians-youtube/blob/add-kriti-s-fellow-folder/fellows/kriti-s/2026-10-18-week4-project-update/README.md) |
| **Week 5** | SWE-CEM-v2 & Clean E1–E7 Ablations | 200-task benchmark, zero-leakage shields, McNemar $p = 1.57 \times 10^{-11}$ validation | [Week 5 Update on GitHub](https://github.com/KritiShahi/humanitarians-youtube/blob/add-kriti-s-fellow-folder/fellows/kriti-s/2026-10-25-week5-project-update/README.md) |


---

## 7. Citation & Attribution

If using or referencing this architecture, benchmark harness, or empirical findings:

```bibtex
@article{shahi2026swecemv2,
  title   = {Cognitive and Contrastive Episodic Memory Architectures for Frozen Language Agents},
  author  = {Shahi, Kriti},
  journal = {Humanitarians AI Inc. Research Program: RL Environment},
  year    = {2026},
  month   = {October},
  note    = {Evaluated on Nebius SWE-rebench defect benchmarks (N=200)}
}
```

**Lead Researcher:** Kriti Shahi ([GitHub: @KritiShahi](https://github.com/KritiShahi))  
**Organization:** Humanitarians AI Inc. — RL Environment Research Group  
**Project:** Cognitive Memory Architectures for Pi Agent  
**GitHub Branch:** [`KritiShahi/humanitarians-youtube (add-kriti-s-fellow-folder)`](https://github.com/KritiShahi/humanitarians-youtube/tree/add-kriti-s-fellow-folder/fellows/kriti-s)
