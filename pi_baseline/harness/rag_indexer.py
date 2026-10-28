#!/usr/bin/env python3
"""
Sliding-Window Chunker & Flat Vector/BM25 Indexer for Arm 1 (Ordinary RAG).
Implements:
  - 256-token sliding window with 32-token overlap.
  - TF-IDF / BM25 and token cosine similarity.
  - Top-k untyped chunk retrieval.
"""

import math
import re
from typing import List, Dict, Any, Tuple

class NaiveRAGIndexer:
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 32):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[Dict[str, Any]] = []

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def add_document(self, doc_id: str, text: str) -> int:
        """Splits document using sliding window and stores raw chunks."""
        words = text.split()
        if not words:
            return 0
            
        step = max(1, self.chunk_size - self.chunk_overlap)
        added = 0
        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            self.chunks.append({
                "chunk_id": f"{doc_id}_chunk_{len(self.chunks)}",
                "doc_id": doc_id,
                "text": chunk_text,
                "tokens": self._tokenize(chunk_text)
            })
            added += 1
            if i + self.chunk_size >= len(words):
                break
        return added

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Computes BM25/cosine relevance and returns top-k untyped chunks."""
        q_tokens = self._tokenize(query)
        if not q_tokens or not self.chunks:
            return []

        # Compute term frequencies
        q_set = set(q_tokens)
        doc_count = len(self.chunks)
        df: Dict[str, int] = {}
        for token in q_set:
            df[token] = sum(1 for c in self.chunks if token in c["tokens"])

        scores: List[Tuple[float, Dict[str, Any]]] = []
        for chunk in self.chunks:
            tokens = chunk["tokens"]
            score = 0.0
            for token in q_tokens:
                tf = tokens.count(token)
                if tf > 0:
                    idf = math.log((doc_count - df.get(token, 0) + 0.5) / (df.get(token, 0) + 0.5) + 1.0)
                    score += (tf * (1.5 + 1.0)) / (tf + 1.5 * (1.0 - 0.75 + 0.75 * (len(tokens) / 50.0))) * idf
            scores.append((score, chunk))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in scores[:top_k] if score > 0]

    def format_untyped_context(self, retrieved: List[Dict[str, Any]], max_tokens: int = 1000) -> str:
        """Formats retrieved chunks as an unstructured text block within budget."""
        if not retrieved:
            return ""
        
        lines = ["=== RELEVANT CONTEXT ==="]
        curr_words = len(lines[0].split())
        for r in retrieved:
            chunk_text = f"[{r['doc_id']}] {r['text']}"
            words_count = len(chunk_text.split())
            if curr_words + words_count <= max_tokens:
                lines.append(chunk_text)
                curr_words += words_count
            else:
                break
        lines.append("=== END CONTEXT ===")
        return "\n".join(lines)
