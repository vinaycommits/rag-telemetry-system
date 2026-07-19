# src/main.py

from src.engine.retriever import HybridRetriever
from src.engine.reranker  import Reranker
from src.engine.generator import Generator
from src.telemetry.logger import TelemetryLogger


class RAGPipeline:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.reranker  = Reranker()
        self.generator = Generator()
        self.logger    = TelemetryLogger()

    def index(self, texts: list[str], sources: list[str]):
        self.retriever.index(texts, sources)

    def query(self, user_query: str) -> dict:

        # Step 1 — Retrieve
        retrieval_result = self.retriever.retrieve(user_query)
        chunks           = retrieval_result["chunks"]

        # Step 2 — Rerank
        chunks = self.reranker.rerank(user_query, chunks)

        # Step 3 — Generate
        output = self.generator.generate(user_query, chunks)

        # Step 4 — Log retrieval
        self.logger.log_retrieval(
            query_id     = retrieval_result["query_id"],
            query        = retrieval_result["query"],
            timestamp    = retrieval_result["timestamp"],
            retrieval_ms = retrieval_result["retrieval_ms"],
            chunks       = chunks,
        )

        # Step 5 — Attribute + log answer
        used_ids = self._attribute(output["answer"], chunks)
        self.logger.log_answer(
            query_id       = retrieval_result["query_id"],
            answer         = output["answer"],
            llm_ms         = output["latency_ms"],
            used_chunk_ids = used_ids,
        )

        # Step 6 — Return everything
        return {
            "query_id": retrieval_result["query_id"],
            "answer":   output["answer"],
            "chunks":   chunks,
            "latency":  {
                "retrieval_ms": retrieval_result["retrieval_ms"],
                "llm_ms":       output["latency_ms"],
            },
        }

    def _attribute(self, answer: str, chunks: list) -> list[str]:
        used = []
        for chunk in chunks:
            snippet = chunk.text[:15].lower()
            if snippet in answer.lower():
                used.append(chunk.chunk_id)
        return used


if __name__ == "__main__":
    texts = [
        "BM25 is a keyword-based ranking function used in search.",
        "FAISS is a library for fast vector similarity search.",
        "RAG grounds LLM outputs in retrieved documents.",
        "Cross-encoders score query-document pairs jointly.",
        "RRF combines ranked lists from multiple search systems.",
    ]
    sources = ["doc1.txt", "doc2.txt", "doc3.txt", "doc4.txt", "doc5.txt"]

    pipeline = RAGPipeline()
    pipeline.index(texts, sources)

    queries = [
        "how does keyword search work?",
        "what is a cross-encoder?",
        "how does RAG reduce hallucination?",
    ]

    for q in queries:
        result = pipeline.query(q)
        print(f"\nQ: {q}")
        print(f"A: {result['answer']}")
        print(f"Retrieval: {result['latency']['retrieval_ms']:.1f}ms")
        print(f"Top chunk: {result['chunks'][0].text[:60]}...")

    print(f"\nChunk utilization: {pipeline.logger.chunk_utilization_rate()}%")
    pipeline.logger.export_jsonl()