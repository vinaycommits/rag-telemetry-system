from sentence_transformers import CrossEncoder
from src.config import RERANK_MODEL , TOP_K_RERANK
from src.engine.retriever import RetrievedChunk

class Reranker:
    def __init__(self):
        print('loading rerank model...')
        self.model=CrossEncoder(RERANK_MODEL)

    def rerank(self,query:str,chunks:list[RetrievedChunk]) -> list[RetrievedChunk]:
        pairs=[(query,chunk.text) for chunk in chunks]

        scores=self.model.predict(pairs)

        for chunk,score in zip(chunks,scores):
            chunk.rerank_score=float(score)

        reranked=sorted(chunks,key=lambda c:c.rerank_score,reverse=True)

        for rank,chunk in enumerate(reranked):
            chunk.rank_after_rerank=rank

        return reranked[:TOP_K_RERANK]
    
# if __name__ == "__main__":
#     import copy
#     from src.engine.retriever import HybridRetriever

#     # Build retriever and index
#     retriever = HybridRetriever()
#     texts = [
#         "BM25 is a keyword-based ranking function used in search.",
#         "FAISS is a library for fast vector similarity search.",
#         "RAG grounds LLM outputs in retrieved documents.",
#         "Cross-encoders score query-document pairs jointly.",
#         "RRF combines ranked lists from multiple search systems.",
#     ]
#     sources = ["doc1.txt", "doc2.txt", "doc3.txt", "doc4.txt", "doc5.txt"]
#     retriever.index(texts, sources)

#     # Retrieve
#     result = retriever.retrieve("how does keyword search work?")
#     chunks = result["chunks"]

#     print("--- BEFORE reranking ---")
#     for c in chunks:
#         print(f"  [{c.rank_before_rerank}] {c.text[:50]}...")

#     # Rerank
#     reranker = Reranker()
#     reranked  = reranker.rerank("how does keyword search work?", chunks)

#     print("\n--- AFTER reranking ---")
#     for c in reranked:
#         print(f"  [{c.rank_after_rerank}] rerank={c.rerank_score:.3f}  {c.text[:50]}...")
