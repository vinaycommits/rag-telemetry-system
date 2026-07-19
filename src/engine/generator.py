# src/engine/generator.py

import os
import time
from groq import Groq
from dotenv import load_dotenv
from src.engine.retriever import RetrievedChunk

load_dotenv()


class Generator:
    def __init__(self):
        self.client     = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = "llama-3.1-8b-instant"

    def _build_prompt(self, query: str, chunks: list[RetrievedChunk]) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(f"[{i+1}] (source: {chunk.source})\n{chunk.text}")
        context = "\n\n".join(context_parts)

        return f"""You are a precise assistant. Answer using ONLY the context below.
If the answer cannot be found in the context, you MUST respond with exactly:
I don't know

Do NOT make up information. Do NOT use your training knowledge.

Context:
{context}

Question: {query}

Answer:"""

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> dict:
        prompt = self._build_prompt(query, chunks)

        t0       = time.perf_counter()
        response = self.client.chat.completions.create(
            model       = self.model_name,
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "answer":     response.choices[0].message.content.strip(),
            "latency_ms": latency_ms,
            "prompt":     prompt,
        }
    
# if __name__ == "__main__":
#     from src.engine.retriever import HybridRetriever
#     from src.engine.reranker  import Reranker

#     texts = [
#         "BM25 is a keyword-based ranking function used in search.",
#         "FAISS is a library for fast vector similarity search.",
#         "RAG grounds LLM outputs in retrieved documents.",
#         "Cross-encoders score query-document pairs jointly.",
#         "RRF combines ranked lists from multiple search systems.",
#     ]
#     sources = ["doc1.txt", "doc2.txt", "doc3.txt", "doc4.txt", "doc5.txt"]

#     retriever = HybridRetriever()
#     retriever.index(texts, sources)

#     query  = "what is BM25?"
#     result = retriever.retrieve(query)
#     chunks = result["chunks"]

#     reranker = Reranker()
#     chunks   = reranker.rerank(query, chunks)

#     generator = Generator()
#     output    = generator.generate(query, chunks)

#     print(f"\nQuery: {query}")
#     print(f"Answer: {output['answer']}")
#     print(f"Latency: {output['latency_ms']:.1f}ms")
#     print(f"\nChunks used as context:")
#     for i, c in enumerate(chunks):
#         print(f"  [{i+1}] {c.text[:60]}...")