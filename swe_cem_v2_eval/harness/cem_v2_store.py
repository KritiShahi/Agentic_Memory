#!/usr/bin/env python3
"""
Contrastive Episodic Memory v2 (CEM-v2) Storage & Retrieval Engine.

Features:
- Schema v2.0.0 supporting:
    - execution_phase: [planning, code_editing, test_execution, recovery, verification]
    - error_category: [TypeError_AttributeError, MissingBoundaryCheck_OffByOne,
                       TestAssertionMockFailure, Import_DependencyError,
                       AsyncDeadlock_RaceCondition, GeneralDefect]
    - tool_affinity: [edit, bash, read, any]
    - constraint_type: [HARD_CONSTRAINT, SOFT_RECOMMENDATION]
    - code_delta_summary: Declarative code difference representation (replaces raw tool traces)
    - state_preconditions_json, active_constraints_json
- Dual-Index Architecture:
    1. Lexical FTS5 index on diagnostics, invariants, prescriptions, and code deltas.
    2. 128-dim L2-normalized dense semantic embeddings.
- Novelty Dampening:
    Score_final = Score_hybrid / (1 + gamma * ln(1 + access_count)) with gamma = 0.15,
    actively breaking template collapse and high-frequency episode monopolies.
- Gated Applicability Filtering:
    Enforces phase, error category, tool affinity, and constraint type matching.
"""

import hashlib
import json
import math
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class DenseEmbeddingEngineV2:
    """Computes a 128-dimensional normalized dense semantic representation."""
    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self.dim

        vec = [0.0] * self.dim
        normalized = re.sub(r"[^\w\s\.\-_/]", " ", text.lower())
        tokens = normalized.split()

        for token in tokens:
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if ((h >> 8) & 1) else -1.0
            vec[idx] += sign * 1.5

            for i in range(len(token) - 2):
                ngram = token[i:i+3]
                nh = int(hashlib.md5(ngram.encode("utf-8")).hexdigest(), 16)
                nidx = nh % self.dim
                nsign = 1.0 if ((nh >> 8) & 1) else -1.0
                vec[nidx] += nsign * 0.5

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        dot = sum(a * b for a, b in zip(vec1, vec2))
        return max(0.0, min(1.0, (dot + 1.0) / 2.0))


class CEMv2MemoryStore:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(Path(__file__).resolve().parent.parent / "memory" / "cem_v2_contrastive_memory.sqlite")
        self.db_path = db_path
        self._mem_conn = None
        if self.db_path == ":memory:":
            self._mem_conn = sqlite3.connect(":memory:")
            self._mem_conn.row_factory = sqlite3.Row
            self._mem_conn.execute("PRAGMA foreign_keys = ON;")

        self.embedder = DenseEmbeddingEngineV2(dim=128)
        self.gamma_novelty = 0.15
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._mem_conn is not None:
            return self._mem_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    schema_version TEXT NOT NULL,
                    problem_signature TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    pi_revision TEXT NOT NULL,
                    origin_session_id TEXT NOT NULL,
                    origin_run_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    repository TEXT NOT NULL,
                    subsystem TEXT NOT NULL,
                    language TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    execution_phase TEXT NOT NULL,
                    error_category TEXT NOT NULL,
                    tool_affinity TEXT NOT NULL,
                    constraint_type TEXT NOT NULL,
                    diagnostic_error TEXT NOT NULL,
                    initial_state_json TEXT NOT NULL,
                    failed_trajectory_json TEXT NOT NULL,
                    successful_trajectory_json TEXT NOT NULL,
                    divergence_point_json TEXT NOT NULL,
                    lesson_invariant TEXT NOT NULL,
                    lesson_prescription TEXT NOT NULL,
                    code_delta_summary TEXT NOT NULL,
                    state_preconditions_json TEXT NOT NULL,
                    active_constraints_json TEXT NOT NULL,
                    tests_passed INTEGER NOT NULL,
                    tests_failed INTEGER NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    last_accessed_at TEXT,
                    success_reinforcements INTEGER NOT NULL DEFAULT 0,
                    failure_penalties INTEGER NOT NULL DEFAULT 0,
                    is_tombstoned INTEGER NOT NULL DEFAULT 0,
                    superseded_by TEXT,
                    embedding_json TEXT NOT NULL,
                    estimated_tokens INTEGER NOT NULL
                );
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_episodes_applicability
                ON episodes (language, execution_phase, error_category, tool_affinity, constraint_type);
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_episodes_lookup
                ON episodes (problem_signature, is_tombstoned, superseded_by);
            """)

            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
                    episode_id UNINDEXED,
                    problem_signature,
                    error_category,
                    execution_phase,
                    diagnostic_error,
                    failed_hypothesis,
                    successful_hypothesis,
                    lesson_invariant,
                    lesson_prescription,
                    code_delta_summary,
                    tokenize='porter unicode61'
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS revision_audit (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata_json TEXT
                );
            """)
            conn.commit()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return max(1, math.ceil(len(text) / 4))

    def write(self, episode_data: Dict[str, Any]) -> str:
        """Atomically stores a verified CEM-v2 episode."""
        verif = episode_data.get("verification", {})
        tests_passed = verif.get("testsPassed", 0)
        tests_failed = verif.get("testsFailed", 0)
        if tests_passed <= 0 or tests_failed != 0:
            raise ValueError(f"Write rejected: Unverified episode (passed={tests_passed}, failed={tests_failed})")

        now_iso = datetime.now(timezone.utc).isoformat()
        episode_id = episode_data.get("episodeId") or f"ep_v2_{hashlib.md5(now_iso.encode()).hexdigest()[:12]}"

        diag_err = episode_data.get("initialState", {}).get("diagnosticError", "")
        prob_sig = episode_data.get("problemSignature") or hashlib.md5(diag_err.encode("utf-8")).hexdigest()[:16]

        prov = episode_data.get("provenance", {})
        scope = episode_data.get("scope", {})
        failed_traj = episode_data.get("failedTrajectory", {})
        succ_traj = episode_data.get("successfulTrajectory", {})
        div_point = episode_data.get("divergencePoint", {})
        lesson = episode_data.get("lesson", {})
        applicability = episode_data.get("applicability", {})

        exec_phase = applicability.get("executionPhase", "planning")
        err_cat = applicability.get("errorCategory", "GeneralDefect")
        tool_aff = applicability.get("toolAffinity", "edit")
        constraint_type = applicability.get("constraintType", "SOFT_RECOMMENDATION")

        code_delta = div_point.get("causalPivotDelta", "")
        full_text_for_embedding = f"{err_cat} {diag_err} {lesson.get('invariantConstraint', '')} {lesson.get('actionablePrescription', '')} {code_delta}"
        embedding = self.embedder.embed(full_text_for_embedding)

        formatted_guidance = (
            f"[CEM-v2 EPISODE: {prob_sig} | Phase: {exec_phase} | Category: {err_cat}]\n"
            f"- Invariant: {lesson.get('invariantConstraint', '')}\n"
            f"- Prescription: {lesson.get('actionablePrescription', '')}\n"
            f"- Code Delta: {code_delta}"
        )
        est_tokens = self.estimate_tokens(formatted_guidance)

        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT episode_id FROM episodes WHERE problem_signature = ? AND execution_phase = ? AND is_tombstoned = 0 AND superseded_by IS NULL",
                (prob_sig, exec_phase)
            )
            existing = cur.fetchone()
            if existing:
                conn.execute(
                    "UPDATE episodes SET superseded_by = ?, updated_at = ? WHERE episode_id = ?",
                    (episode_id, now_iso, existing["episode_id"])
                )

            conn.execute("""
                INSERT INTO episodes (
                    episode_id, schema_version, problem_signature, model_id, pi_revision,
                    origin_session_id, origin_run_id, created_at, updated_at,
                    repository, subsystem, language, task_type,
                    execution_phase, error_category, tool_affinity, constraint_type,
                    diagnostic_error, initial_state_json,
                    failed_trajectory_json, successful_trajectory_json,
                    divergence_point_json, lesson_invariant, lesson_prescription,
                    code_delta_summary, state_preconditions_json, active_constraints_json,
                    tests_passed, tests_failed, access_count, last_accessed_at,
                    success_reinforcements, failure_penalties, is_tombstoned,
                    superseded_by, embedding_json, estimated_tokens
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, 0, 0, 0, NULL, ?, ?);
            """, (
                episode_id,
                episode_data.get("schemaVersion", "2.0.0"),
                prob_sig,
                prov.get("modelId", "glm-5.3-flash"),
                prov.get("piRevision", "cem-v2-eval"),
                prov.get("originSessionId", "session_v2"),
                prov.get("originRunId", "run_v2"),
                prov.get("createdAt", now_iso),
                now_iso,
                scope.get("repository", "unknown_repo"),
                scope.get("subsystem", "default"),
                scope.get("language", "python"),
                scope.get("taskType", "github_issue_fix"),
                exec_phase,
                err_cat,
                tool_aff,
                constraint_type,
                diag_err,
                json.dumps(episode_data.get("initialState", {})),
                json.dumps(failed_traj),
                json.dumps(succ_traj),
                json.dumps(div_point),
                lesson.get("invariantConstraint", ""),
                lesson.get("actionablePrescription", ""),
                code_delta,
                json.dumps(applicability.get("statePreconditions", {})),
                json.dumps(applicability.get("activeConstraints", [])),
                tests_passed,
                tests_failed,
                json.dumps(embedding),
                est_tokens
            ))

            # FTS5 insert
            conn.execute("""
                INSERT INTO episodes_fts (
                    episode_id, problem_signature, error_category, execution_phase,
                    diagnostic_error, failed_hypothesis, successful_hypothesis,
                    lesson_invariant, lesson_prescription, code_delta_summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                episode_id,
                prob_sig,
                err_cat,
                exec_phase,
                diag_err,
                failed_traj.get("hypothesis", ""),
                succ_traj.get("hypothesis", ""),
                lesson.get("invariantConstraint", ""),
                lesson.get("actionablePrescription", ""),
                code_delta
            ))

            conn.execute("""
                INSERT INTO revision_audit (episode_id, action, reason, timestamp, metadata_json)
                VALUES (?, 'WRITE_CREATED', 'CEM-v2 verified episode harvest', ?, ?);
            """, (episode_id, now_iso, json.dumps({"phase": exec_phase, "error_category": err_cat})))
            conn.commit()

        return episode_id

    def retrieve_with_applicability(
        self,
        query_text: str,
        diagnostic_error: str = "",
        execution_phase: str = "planning",
        error_category: Optional[str] = None,
        tool_affinity: str = "any",
        constraint_type: Optional[str] = None,
        scope: Optional[Dict[str, str]] = None,
        top_k: int = 3,
        max_tokens: int = 1000,
        score_threshold: float = 0.68,
        apply_novelty_dampening: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Executes phased, applicability-gated retrieval with Novelty Dampening.
        """
        combined_query = f"{error_category or ''} {diagnostic_error} {query_text}".strip()
        query_vec = self.embedder.embed(combined_query)
        now_iso = datetime.now(timezone.utc).isoformat()

        with self._get_connection() as conn:
            # 1. SQL filtering based on applicability
            sql = "SELECT * FROM episodes WHERE is_tombstoned = 0 AND superseded_by IS NULL"
            params: List[Any] = []

            if scope and "language" in scope:
                sql += " AND language = ?"
                params.append(scope["language"])

            if execution_phase and execution_phase != "any":
                sql += " AND (execution_phase = ? OR execution_phase = 'any')"
                params.append(execution_phase)

            if error_category and error_category != "GeneralDefect":
                sql += " AND (error_category = ? OR error_category = 'GeneralDefect')"
                params.append(error_category)

            if tool_affinity and tool_affinity != "any":
                sql += " AND (tool_affinity = ? OR tool_affinity = 'any')"
                params.append(tool_affinity)

            if constraint_type:
                sql += " AND constraint_type = ?"
                params.append(constraint_type)

            cur = conn.execute(sql, params)
            rows = cur.fetchall()

            # Fallback relaxation if no strict matches found
            if not rows and error_category:
                fallback_sql = "SELECT * FROM episodes WHERE is_tombstoned = 0 AND superseded_by IS NULL AND (execution_phase = ? OR execution_phase = 'any')"
                cur = conn.execute(fallback_sql, (execution_phase,))
                rows = cur.fetchall()

            if not rows:
                return []

            # 2. Lexical scoring via FTS5
            fts_scores: Dict[str, float] = {}
            if combined_query:
                clean_terms = [w for w in re.findall(r"\w+", combined_query) if len(w) > 2 and w.lower() not in {"str_replace_editor", "bash", "pytest", "view", "workspace", "action"}]
                if clean_terms:
                    fts_query = " OR ".join(clean_terms[:8])
                    try:
                        fts_cur = conn.execute(
                            "SELECT episode_id, rank FROM episodes_fts WHERE episodes_fts MATCH ? ORDER BY rank",
                            (fts_query,)
                        )
                        fts_rows = fts_cur.fetchall()
                        if fts_rows:
                            max_mag = max(abs(float(r["rank"])) for r in fts_rows)
                            if max_mag > 0:
                                for fts_row in fts_rows:
                                    fts_scores[fts_row["episode_id"]] = abs(float(fts_row["rank"])) / max_mag
                    except sqlite3.OperationalError:
                        pass

            # 3. Hybrid ranking + Novelty Dampening
            candidates = []
            for row in rows:
                ep_id = row["episode_id"]
                ep_vec = json.loads(row["embedding_json"])
                dense_sim = self.embedder.cosine_similarity(query_vec, ep_vec)
                bm25_sim = fts_scores.get(ep_id, 0.0)

                hybrid_score = (0.65 * dense_sim) + (0.35 * bm25_sim)

                # Penalize known failures
                if row["failure_penalties"] > 0:
                    hybrid_score *= max(0.5, 1.0 - (row["failure_penalties"] * 0.2))

                # Apply Novelty Dampening
                if apply_novelty_dampening:
                    access_cnt = row["access_count"]
                    dampening_mult = 1.0 / (1.0 + self.gamma_novelty * math.log(1.0 + access_cnt))
                    final_score = hybrid_score * dampening_mult
                else:
                    final_score = hybrid_score

                if final_score >= score_threshold:
                    candidates.append((final_score, row))

            candidates.sort(key=lambda x: x[0], reverse=True)

            # 4. Token allocation
            results = []
            accumulated_tokens = 0
            accessed_ids = []

            for score, row in candidates:
                if len(results) >= top_k:
                    break
                ep_tokens = row["estimated_tokens"]
                if accumulated_tokens + ep_tokens > max_tokens:
                    continue

                accumulated_tokens += ep_tokens
                accessed_ids.append(row["episode_id"])

                results.append({
                    "episodeId": row["episode_id"],
                    "problemSignature": row["problem_signature"],
                    "score": round(score, 4),
                    "executionPhase": row["execution_phase"],
                    "errorCategory": row["error_category"],
                    "toolAffinity": row["tool_affinity"],
                    "constraintType": row["constraint_type"],
                    "scope": {
                        "repository": row["repository"],
                        "subsystem": row["subsystem"],
                        "language": row["language"],
                        "taskType": row["task_type"]
                    },
                    "diagnosticError": row["diagnostic_error"],
                    "lesson": {
                        "invariantConstraint": row["lesson_invariant"],
                        "actionablePrescription": row["lesson_prescription"]
                    },
                    "codeDeltaSummary": row["code_delta_summary"],
                    "divergencePoint": json.loads(row["divergence_point_json"]),
                    "statePreconditions": json.loads(row["state_preconditions_json"]),
                    "activeConstraints": json.loads(row["active_constraints_json"]),
                    "estimatedTokens": ep_tokens,
                    "accessCount": row["access_count"]
                })

            if accessed_ids:
                placeholders = ",".join("?" * len(accessed_ids))
                conn.execute(
                    f"UPDATE episodes SET access_count = access_count + 1, last_accessed_at = ? WHERE episode_id IN ({placeholders})",
                    [now_iso] + accessed_ids
                )
                conn.commit()

            return results
