# src/telemetry/logger.py

import json
import sqlite3
import threading
from pathlib import Path
from datetime import datetime, timezone
from src.engine.retriever import RetrievedChunk


DB_PATH = Path("database/telemetry.db")
DB_PATH.parent.mkdir(exist_ok=True)

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _create_tables(_local.conn)
    return _local.conn


def _create_tables(conn: sqlite3.Connection):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS queries (
        query_id        TEXT PRIMARY KEY,
        query_text      TEXT NOT NULL,
        timestamp       TEXT NOT NULL,
        retrieval_ms    REAL,
        rerank_ms       REAL,
        llm_ms          REAL,
        final_answer    TEXT
    );

    CREATE TABLE IF NOT EXISTS chunks (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        query_id            TEXT NOT NULL,
        chunk_id            TEXT NOT NULL,
        source              TEXT,
        text_snippet        TEXT,
        bm25_score          REAL,
        vector_score        REAL,
        hybrid_score        REAL,
        rerank_score        REAL,
        rank_before_rerank  INTEGER,
        rank_after_rerank   INTEGER,
        used_in_answer      INTEGER,
        FOREIGN KEY (query_id) REFERENCES queries(query_id)
    );
    """)
    conn.commit()

class TelemetryLogger:

    def log_retrieval(
        self,
        query_id:     str,
        query:        str,
        timestamp:    str,
        retrieval_ms: float,
        chunks:       list[RetrievedChunk],
    ):
        conn = _get_conn()

        # Save the query
        conn.execute(
            """INSERT OR IGNORE INTO queries
               (query_id, query_text, timestamp, retrieval_ms)
               VALUES (?, ?, ?, ?)""",
            (query_id, query, timestamp, retrieval_ms),
        )

        # Save every chunk with all its scores
        for chunk in chunks:
            conn.execute(
                """INSERT INTO chunks
                   (query_id, chunk_id, source, text_snippet,
                    bm25_score, vector_score, hybrid_score,
                    rerank_score, rank_before_rerank, rank_after_rerank)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    query_id,
                    chunk.chunk_id,
                    chunk.source,
                    chunk.text[:300],
                    chunk.bm25_score,
                    chunk.vector_score,
                    chunk.hybrid_score,
                    chunk.rerank_score,
                    chunk.rank_before_rerank,
                    chunk.rank_after_rerank,
                ),
            )
        conn.commit()

    def log_answer(
        self,
        query_id:   str,
        answer:     str,
        llm_ms:     float,
        used_chunk_ids: list[str],
    ):
        conn = _get_conn()

        # Save the answer and latency
        conn.execute(
            """UPDATE queries
               SET final_answer=?, llm_ms=?
               WHERE query_id=?""",
            (answer, llm_ms, query_id),
        )

        # Mark which chunks were actually used
        for chunk_id in used_chunk_ids:
            conn.execute(
                """UPDATE chunks SET used_in_answer=1
                   WHERE query_id=? AND chunk_id=?""",
                (query_id, chunk_id),
            )

        # Mark the rest as not used
        conn.execute(
            """UPDATE chunks SET used_in_answer=0
               WHERE query_id=? AND used_in_answer IS NULL""",
            (query_id,),
        )
        conn.commit()

    def chunk_utilization_rate(self) -> float:
        """What % of retrieved chunks actually got used in answers?"""
        conn = _get_conn()
        row  = conn.execute(
            "SELECT AVG(used_in_answer) FROM chunks WHERE used_in_answer IS NOT NULL"
        ).fetchone()
        return round((row[0] or 0) * 100, 2)

    def export_jsonl(self, path: str = "database/traces.jsonl"):
        """Export every query + its chunks as JSONL for offline analysis."""
        conn    = _get_conn()
        queries = conn.execute("SELECT * FROM queries").fetchall()

        with open(path, "w") as f:
            for q in queries:
                q = dict(q)
                chunks = conn.execute(
                    "SELECT * FROM chunks WHERE query_id=?",
                    (q["query_id"],)
                ).fetchall()
                q["chunks"] = [dict(c) for c in chunks]
                f.write(json.dumps(q) + "\n")

        print(f"Exported {len(queries)} traces to {path}")

# if __name__ == "__main__":
#     from src.engine.retriever import HybridRetriever
#     from src.engine.reranker  import Reranker
#     from src.engine.generator import Generator

#     # --- Setup ---
#     texts = [
#         "BM25 is a keyword-based ranking function used in search.",
#         "FAISS is a library for fast vector similarity search.",
#         "RAG grounds LLM outputs in retrieved documents.",
#         "Cross-encoders score query-document pairs jointly.",
#         "RRF combines ranked lists from multiple search systems.",
#     ]
#     sources   = ["doc1.txt", "doc2.txt", "doc3.txt", "doc4.txt", "doc5.txt"]

#     retriever = HybridRetriever()
#     retriever.index(texts, sources)
#     reranker  = Reranker()
#     generator = Generator()
#     logger    = TelemetryLogger()

#     # --- Run pipeline ---
#     query  = "how does keyword search work?"
#     result = retriever.retrieve(query)
#     chunks = reranker.rerank(query, result["chunks"])
#     output = generator.generate(query, chunks)

#     # --- Log retrieval ---
#     logger.log_retrieval(
#         query_id     = result["query_id"],
#         query        = result["query"],
#         timestamp    = result["timestamp"],
#         retrieval_ms = result["retrieval_ms"],
#         chunks       = chunks,
#     )

#     # --- Log answer ---
#     # For now we say the top chunk was used (stub attribution)
#     used_ids = [chunks[0].chunk_id]
#     logger.log_answer(
#         query_id       = result["query_id"],
#         answer         = output["answer"],
#         llm_ms         = output["latency_ms"],
#         used_chunk_ids = used_ids,
#     )

#     # --- Check stats ---
#     print(f"Chunk utilization rate: {logger.chunk_utilization_rate()}%")
#     logger.export_jsonl()