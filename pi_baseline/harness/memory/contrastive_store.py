#!/usr/bin/env python3
"""
Contrastive Episodic Memory Storage Engine (Architecture 1).

Features:
- SQLite WAL mode for high-concurrency ACID transactions.
- Dual-Index Architecture:
    1. Lexical Inverted Index: SQLite FTS5 index on problem signatures, diagnostics, and lessons.
    2. Dense Vector Index: Deterministic normalized dense semantic embeddings with cosine similarity.
- Formal CRUD Lifecycle:
    - write(): Validates schema, hashes diagnostic signatures, inserts into relational + FTS5 + vector stores.
    - retrieve(): Hybrid ranking (0.65*Dense + 0.35*BM25), score threshold filtering (theta >= 0.72),
                 strict token budgeting (B_ctx <= 1000 tokens, top-3 max).
    - revise(): Supersedes prior episodes, archives lineage in revision_audit, and updates active pointers.
    - forget(): Supports explicit tombstoning and Negative Utility Pruning (auto-tombstone at failure_penalties >= 2).
- Stale & Conflicting Memory Handling: Resolves environment drift conflicts via AST snapshot matching and recency.
"""

import json
import math
import os
import re
import sqlite3
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class DenseEmbeddingEngine:
    """
    Deterministic dense semantic embedding generator.
    Computes a 128-dimensional L2-normalized feature representation from text
    using character n-grams and subword token hashing.
    """
    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self.dim
            
        vec = [0.0] * self.dim
        normalized = re.sub(r"[^\w\s\.\-_/]", " ", text.lower())
        tokens = normalized.split()
        
        # Word and subword n-gram hashing
        for token in tokens:
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if ((h >> 8) & 1) else -1.0
            vec[idx] += sign * 1.5
            
            # Character trigrams for morphological and error code matching (e.g., TS2835, ERR_MODULE)
            for i in range(len(token) - 2):
                ngram = token[i:i+3]
                nh = int(hashlib.md5(ngram.encode("utf-8")).hexdigest(), 16)
                nidx = nh % self.dim
                nsign = 1.0 if ((nh >> 8) & 1) else -1.0
                vec[nidx] += nsign * 0.5
                
        # L2 Normalization
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        dot = sum(a * b for a, b in zip(vec1, vec2))
        return max(0.0, min(1.0, (dot + 1.0) / 2.0))  # Scale to [0, 1]


class ContrastiveMemoryStore:
    """
    SQLite-backed storage and retrieval adapter for Contrastive Episodic Memory.
    """
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            baseline_dir = Path(__file__).resolve().parent.parent.parent
            db_path = str(baseline_dir / "episodic_memory.sqlite")
            
        self.db_path = db_path
        self._mem_conn = None
        if self.db_path == ":memory:":
            self._mem_conn = sqlite3.connect(":memory:")
            self._mem_conn.row_factory = sqlite3.Row
            self._mem_conn.execute("PRAGMA foreign_keys = ON;")
            
        self.embedder = DenseEmbeddingEngine(dim=128)
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
        """Initializes relational tables, FTS5 full-text index, and audit trail."""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        with self._get_connection() as conn:
            # 1. Main Episodes Table
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
                    diagnostic_error TEXT NOT NULL,
                    initial_state_json TEXT NOT NULL,
                    failed_trajectory_json TEXT NOT NULL,
                    successful_trajectory_json TEXT NOT NULL,
                    divergence_point_json TEXT NOT NULL,
                    lesson_invariant TEXT NOT NULL,
                    lesson_prescription TEXT NOT NULL,
                    tests_passed INTEGER NOT NULL,
                    tests_failed INTEGER NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed_at TEXT,
                    success_reinforcements INTEGER DEFAULT 0,
                    failure_penalties INTEGER DEFAULT 0,
                    is_tombstoned INTEGER DEFAULT 0,
                    superseded_by TEXT,
                    embedding_json TEXT NOT NULL,
                    estimated_tokens INTEGER NOT NULL
                );
            """)

            # 2. Secondary Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_problem_sig ON episodes(problem_signature);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_scope ON episodes(repository, language, task_type);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_lifecycle ON episodes(is_tombstoned, superseded_by);")

            # 3. FTS5 Lexical Search Table
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
                    episode_id UNINDEXED,
                    problem_signature,
                    diagnostic_error,
                    failed_hypothesis,
                    successful_hypothesis,
                    lesson_invariant,
                    lesson_prescription,
                    causal_pivot_delta,
                    tokenize = 'porter unicode61'
                );
            """)

            # 4. Revision & Lifecycle Audit Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS revision_audit (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    reason TEXT,
                    timestamp TEXT NOT NULL,
                    metadata_json TEXT
                );
            """)
            conn.commit()

    @staticmethod
    def hash_diagnostic(text: str) -> str:
        """Generates a normalized hash signature for a diagnostic error."""
        normalized = re.sub(r"/\S+/", "", text)  # Strip local path prefixes
        normalized = re.sub(r"\s+", " ", normalized).strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Heuristic token estimator (avg 4 chars per token)."""
        return max(1, math.ceil(len(text) / 4))

    def write(self, episode_data: Dict[str, Any]) -> str:
        """
        Validates and atomically writes a Contrastive Episode.
        Enforces schema invariants:
          - Verification pass assertions (tests_passed > 0 and tests_failed == 0).
          - Schema compliance.
        """
        # 1. Validation
        verif = episode_data.get("verification", {})
        tests_passed = verif.get("testsPassed", 0)
        tests_failed = verif.get("testsFailed", 0)
        if tests_passed <= 0 or tests_failed != 0:
            raise ValueError(f"Write rejected: Unverified episode (passed={tests_passed}, failed={tests_failed})")

        now_iso = datetime.now(timezone.utc).isoformat()
        episode_id = episode_data.get("episodeId") or f"ep_{hashlib.md5(now_iso.encode()).hexdigest()[:12]}"
        
        diag_err = episode_data.get("initialState", {}).get("diagnosticError", "")
        prob_sig = episode_data.get("problemSignature") or self.hash_diagnostic(diag_err)

        prov = episode_data.get("provenance", {})
        scope = episode_data.get("scope", {})
        failed_traj = episode_data.get("failedTrajectory", {})
        succ_traj = episode_data.get("successfulTrajectory", {})
        div_point = episode_data.get("divergencePoint", {})
        lesson = episode_data.get("lesson", {})

        # 2. Compute Embedding & Token footprint
        full_text_for_embedding = f"{diag_err} {failed_traj.get('hypothesis', '')} {succ_traj.get('hypothesis', '')} {lesson.get('invariantConstraint', '')} {lesson.get('actionablePrescription', '')}"
        embedding = self.embedder.embed(full_text_for_embedding)
        
        formatted_guidance = (
            f"[CONTRASTIVE EPISODE: {prob_sig}]\n"
            f"- Invariant Warning: {lesson.get('invariantConstraint', '')}\n"
            f"- Actionable Fix: {lesson.get('actionablePrescription', '')}\n"
            f"- Causal Pivot Delta: {div_point.get('causalPivotDelta', '')}\n"
            f"- Evidence: {tests_passed} passed, {tests_failed} failed."
        )
        est_tokens = self.estimate_tokens(formatted_guidance)

        with self._get_connection() as conn:
            # Check duplicate active episode with identical signature
            cur = conn.execute(
                "SELECT episode_id FROM episodes WHERE problem_signature = ? AND is_tombstoned = 0 AND superseded_by IS NULL",
                (prob_sig,)
            )
            existing = cur.fetchone()
            if existing:
                # Mark prior duplicate as superseded
                conn.execute(
                    "UPDATE episodes SET superseded_by = ?, updated_at = ? WHERE episode_id = ?",
                    (episode_id, now_iso, existing["episode_id"])
                )

            # Insert into main episodes table
            conn.execute("""
                INSERT INTO episodes (
                    episode_id, schema_version, problem_signature, model_id, pi_revision,
                    origin_session_id, origin_run_id, created_at, updated_at,
                    repository, subsystem, language, task_type,
                    diagnostic_error, initial_state_json,
                    failed_trajectory_json, successful_trajectory_json,
                    divergence_point_json, lesson_invariant, lesson_prescription,
                    tests_passed, tests_failed, access_count, last_accessed_at,
                    success_reinforcements, failure_penalties, is_tombstoned,
                    superseded_by, embedding_json, estimated_tokens
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, 0, 0, 0, NULL, ?, ?);
            """, (
                episode_id,
                episode_data.get("schemaVersion", "1.0.0"),
                prob_sig,
                prov.get("modelId", "zhipu/glm-5.3-flash"),
                prov.get("piRevision", "11894012dd461232eb075bc890538b6866860a10"),
                prov.get("originSessionId", "unknown_session"),
                prov.get("originRunId", "unknown_run"),
                prov.get("createdAt", now_iso),
                now_iso,
                scope.get("repository", "unknown_repo"),
                scope.get("subsystem", "unknown_subsystem"),
                scope.get("language", "typescript"),
                scope.get("taskType", "compilation_fix"),
                diag_err,
                json.dumps(episode_data.get("initialState", {})),
                json.dumps(failed_traj),
                json.dumps(succ_traj),
                json.dumps(div_point),
                lesson.get("invariantConstraint", ""),
                lesson.get("actionablePrescription", ""),
                tests_passed,
                tests_failed,
                json.dumps(embedding),
                est_tokens
            ))

            # Insert into FTS5 index
            conn.execute("""
                INSERT INTO episodes_fts (
                    episode_id, problem_signature, diagnostic_error,
                    failed_hypothesis, successful_hypothesis,
                    lesson_invariant, lesson_prescription, causal_pivot_delta
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                episode_id,
                prob_sig,
                diag_err,
                failed_traj.get("hypothesis", ""),
                succ_traj.get("hypothesis", ""),
                lesson.get("invariantConstraint", ""),
                lesson.get("actionablePrescription", ""),
                div_point.get("causalPivotDelta", "")
            ))

            # Record in revision audit
            conn.execute("""
                INSERT INTO revision_audit (episode_id, action, reason, timestamp, metadata_json)
                VALUES (?, 'WRITE_CREATED', 'Initial verified contrastive episode harvest', ?, ?);
            """, (episode_id, now_iso, json.dumps({"tests_passed": tests_passed})))
            
            conn.commit()

        return episode_id

    def retrieve(
        self,
        query_text: str,
        diagnostic_error: str = "",
        scope: Optional[Dict[str, str]] = None,
        top_k: int = 3,
        max_tokens: int = 1000,
        score_threshold: float = 0.72
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid retrieval (0.65*Dense + 0.35*BM25) with strict token budgeting.
        """
        combined_query = f"{diagnostic_error} {query_text}".strip()
        query_vec = self.embedder.embed(combined_query)
        now_iso = datetime.now(timezone.utc).isoformat()

        with self._get_connection() as conn:
            # 1. Fetch all active (non-tombstoned, non-superseded) episodes
            query = "SELECT * FROM episodes WHERE is_tombstoned = 0 AND superseded_by IS NULL"
            params = []
            
            if scope:
                if "language" in scope:
                    query += " AND language = ?"
                    params.append(scope["language"])
                if "taskType" in scope:
                    query += " AND task_type = ?"
                    params.append(scope["taskType"])

            cur = conn.execute(query, params)
            rows = cur.fetchall()
            if not rows:
                return []

            # 2. Lexical scoring via FTS5 if possible
            fts_scores: Dict[str, float] = {}
            if combined_query:
                # Sanitize query for FTS5
                clean_terms = re.findall(r"\w+", combined_query)
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
                        pass  # Fallback to pure vector if FTS syntax edge case occurs

            # 3. Hybrid Ranking
            candidates = []
            for row in rows:
                ep_id = row["episode_id"]
                ep_vec = json.loads(row["embedding_json"])
                dense_sim = self.embedder.cosine_similarity(query_vec, ep_vec)
                bm25_sim = fts_scores.get(ep_id, 0.0)

                # Hybrid score calculation
                hybrid_score = (0.65 * dense_sim) + (0.35 * bm25_sim)

                # Failure penalty discount
                penalties = row["failure_penalties"]
                if penalties > 0:
                    hybrid_score *= max(0.5, 1.0 - (penalties * 0.2))

                if hybrid_score >= score_threshold:
                    candidates.append((hybrid_score, row))

            # 4. Sort descending by score
            candidates.sort(key=lambda x: x[0], reverse=True)

            # 5. Greedy Token Budget Allocation (B_ctx <= max_tokens, count <= top_k)
            results = []
            accumulated_tokens = 0
            accessed_ids = []

            for score, row in candidates:
                if len(results) >= top_k:
                    break
                ep_tokens = row["estimated_tokens"]
                if accumulated_tokens + ep_tokens > max_tokens:
                    continue  # Skip if exceeds token ceiling

                accumulated_tokens += ep_tokens
                accessed_ids.append(row["episode_id"])
                
                results.append({
                    "episodeId": row["episode_id"],
                    "problemSignature": row["problem_signature"],
                    "hybridScore": round(score, 4),
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
                    "divergencePoint": json.loads(row["divergence_point_json"]),
                    "failedTrajectory": json.loads(row["failed_trajectory_json"]),
                    "successfulTrajectory": json.loads(row["successful_trajectory_json"]),
                    "verification": {
                        "testsPassed": row["tests_passed"],
                        "testsFailed": row["tests_failed"]
                    },
                    "estimatedTokens": ep_tokens,
                    "provenance": {
                        "originSessionId": row["origin_session_id"],
                        "createdAt": row["created_at"]
                    }
                })

            # 6. Update access statistics
            if accessed_ids:
                placeholders = ",".join("?" * len(accessed_ids))
                conn.execute(
                    f"UPDATE episodes SET access_count = access_count + 1, last_accessed_at = ? WHERE episode_id IN ({placeholders})",
                    [now_iso] + accessed_ids
                )
                conn.commit()

            return results

    def revise(self, prior_episode_id: str, new_episode_data: Dict[str, Any], reason: str = "") -> str:
        """
        Revises a prior episode by creating a new version and linking lineage via supersededBy.
        """
        new_episode_id = self.write(new_episode_data)
        now_iso = datetime.now(timezone.utc).isoformat()
        
        revision_reason = f"{reason} - Superseded by {new_episode_id}" if reason else f"Superseded by {new_episode_id}"
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE episodes SET superseded_by = ?, updated_at = ? WHERE episode_id = ?",
                (new_episode_id, now_iso, prior_episode_id)
            )
            conn.execute("""
                INSERT INTO revision_audit (episode_id, action, reason, timestamp, metadata_json)
                VALUES (?, 'SUPERSEDED', ?, ?, ?);
            """, (prior_episode_id, revision_reason, now_iso, json.dumps({"new_episode_id": new_episode_id})))
            conn.commit()

        return new_episode_id

    def forget(self, episode_id: str, reason: str = "manual_tombstone") -> bool:
        """
        Soft-deletes (tombstones) an episode so it is excluded from active inference.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cur = conn.execute(
                "UPDATE episodes SET is_tombstoned = 1, updated_at = ? WHERE episode_id = ? AND is_tombstoned = 0",
                (now_iso, episode_id)
            )
            if cur.rowcount > 0:
                conn.execute("""
                    INSERT INTO revision_audit (episode_id, action, reason, timestamp, metadata_json)
                    VALUES (?, 'TOMBSTONED', ?, ?, NULL);
                """, (episode_id, reason, now_iso))
                conn.commit()
                return True
        return False

    def record_feedback(self, episode_id: str, success: bool, penalty_threshold: int = 2) -> Dict[str, Any]:
        """
        Records reinforcement feedback or failure penalties.
        Implements Negative Utility Pruning: Auto-tombstones when failure_penalties >= penalty_threshold.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            if success:
                conn.execute(
                    "UPDATE episodes SET success_reinforcements = success_reinforcements + 1, updated_at = ? WHERE episode_id = ?",
                    (now_iso, episode_id)
                )
                action = "REINFORCED_SUCCESS"
            else:
                conn.execute(
                    "UPDATE episodes SET failure_penalties = failure_penalties + 1, updated_at = ? WHERE episode_id = ?",
                    (now_iso, episode_id)
                )
                action = "PENALIZED_FAILURE"

            # Check if threshold reached for automatic pruning
            cur = conn.execute("SELECT failure_penalties, success_reinforcements, is_tombstoned FROM episodes WHERE episode_id = ?", (episode_id,))
            row = cur.fetchone()
            if not row:
                raise KeyError(f"Episode {episode_id} not found.")

            auto_tombstoned = False
            if row["failure_penalties"] >= penalty_threshold and row["is_tombstoned"] == 0:
                conn.execute(
                    "UPDATE episodes SET is_tombstoned = 1, updated_at = ? WHERE episode_id = ?",
                    (now_iso, episode_id)
                )
                conn.execute("""
                    INSERT INTO revision_audit (episode_id, action, reason, timestamp, metadata_json)
                    VALUES (?, 'AUTO_PRUNED_NEGATIVE_UTILITY', 'Failure penalty threshold reached', ?, ?);
                """, (episode_id, now_iso, json.dumps({"penalties": row["failure_penalties"]})))
                auto_tombstoned = True

            conn.execute("""
                INSERT INTO revision_audit (episode_id, action, reason, timestamp, metadata_json)
                VALUES (?, ?, 'Feedback recorded from evaluation run', ?, ?);
            """, (episode_id, action, now_iso, json.dumps({"success": success})))
            
            conn.commit()
            
            return {
                "episodeId": episode_id,
                "successReinforcements": row["success_reinforcements"],
                "failurePenalties": row["failure_penalties"],
                "isTombstoned": bool(row["is_tombstoned"] or auto_tombstoned)
            }

    def resolve_conflicts(self, diagnostic_error: str, current_env_snapshot: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Resolves conflicting memories by evaluating environment snapshot compatibility,
        recency, and verification strength.
        """
        prob_sig = self.hash_diagnostic(diagnostic_error)
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM episodes WHERE problem_signature = ? AND is_tombstoned = 0 ORDER BY created_at DESC",
                (prob_sig,)
            )
            rows = cur.fetchall()
            if not rows:
                return []

            scored_conflicts = []
            for row in rows:
                env_stored = json.loads(row["initial_state_json"]).get("environmentSnapshot", {})
                # Environment match score
                matching_keys = sum(1 for k, v in current_env_snapshot.items() if env_stored.get(k) == v)
                total_keys = max(1, len(current_env_snapshot))
                env_match_ratio = matching_keys / total_keys

                score = (0.5 * env_match_ratio) + (0.3 * (row["tests_passed"] / (row["tests_passed"] + row["tests_failed"] + 1)))
                if row["superseded_by"] is None:
                    score += 0.2  # Bonus for active version

                scored_conflicts.append((score, dict(row)))

            scored_conflicts.sort(key=lambda x: x[0], reverse=True)
            return [x[1] for x in scored_conflicts]


if __name__ == "__main__":
    store = ContrastiveMemoryStore(":memory:")
    print("ContrastiveMemoryStore initialized in memory successfully.")
