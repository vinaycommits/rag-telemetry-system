import uuid
import time
import numpy as np
import faiss
from dataclasses import dataclass, field
from typing import Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import copy
# import faiss
from src.config import (
    EMBEDDING_MODEL,
    RERANK_MODEL,
    TOP_K_RETRIEVE,
    TOP_K_RERANK,
    BM25_WEIGHT,
    VECTOR_WEIGHT,
    RRF_K,
)

@dataclass
class RetrievedChunk:
    chunk_id:str
    text:str
    source:str
    bm25_score:Optional[float]=None
    vector_score:Optional[float]=None
    hybrid_score:Optional[float]=None
    rerank_score:Optional[float]=None
    rank_before_rerank:Optional[int]=None
    rank_after_rerank:Optional[int]=None
    used_in_answer:Optional[bool]=None

class HybridRetriever:
        def __init__(self):
            print("loading embedding model...")
            self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
            self.chunks: list[RetrievedChunk] = []
            self.bm25: Optional[BM25Okapi] = None
            self.fiass_index: Optional[faiss.IndexFlatIP] = None

        def index(self,texts:list[str],source:list[str]):
            print(f"Indexing {len(texts)} chunks...")

             #1. create chunk objects
            self.chunks = [
                  RetrievedChunk (
                       chunk_id=str(uuid.uuid4()),
                       text = t,
                       source = s,
                  )
                  for t,s in zip(texts,source)
             ] 

            #2 build BM25 index
            tokenized = [t.lower().split() for t in texts]
            self.bm25=BM25Okapi(tokenized)

            embeddings=self.embed_model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=True,
            ).astype("float32")

            dim=embeddings.shape[1]
            self.faiss_index=faiss.IndexFlatIP(dim)
            self.faiss_index.add(embeddings)

            print(f"Done. {len(self.chunks)} chunks indexed.")

        def _bm25_search(self, query: str) -> list[tuple[int, float]]:
           scores  = self.bm25.get_scores(query.lower().split())
           top_idx = np.argsort(scores)[::-1][:TOP_K_RETRIEVE]
           return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0.0]
        
        def _vector_search(self, query: str) -> list[tuple[int, float]]:
           q_emb = self.embed_model.encode(
           [query], normalize_embeddings=True
           ).astype("float32")
           sims, idxs = self.faiss_index.search(q_emb, TOP_K_RETRIEVE)
           return [
           (int(i), float(s))
           for i, s in zip(idxs[0], sims[0])
           if i != -1 and s > -1.0        # -1 index and huge negative = FAISS padding
           ]
        
        def _rrf_fusion(
                  self,
                  bm25_results: list[tuple[int,float]],
                  vector_results: list[tuple[int,float]],
        ) -> list[tuple[int,float]]:
             scores:dict[int,float]={}

             for rank , (idx,_) in enumerate(bm25_results):
                  scores[idx]=scores.get(idx,0.0) + BM25_WEIGHT / (rank+RRF_K)


             for rank , (idx,_) in enumerate(vector_results):
                  scores[idx]=scores.get(idx,0.0) + VECTOR_WEIGHT / (rank+RRF_K)

             return sorted(scores.items() , key=lambda x:x[1],reverse=True)[:TOP_K_RETRIEVE]
        
        def retrieve(self , query:str) -> dict:
             from datetime import datetime , timezone

             t0=time.perf_counter()

             bm25_results=self._bm25_search(query)
             vector_results=self._vector_search(query)
             fused=self._rrf_fusion(bm25_results,vector_results)

             retrieval_ms=(time.perf_counter() - t0) * 1000

             bm25_map=dict(bm25_results)
             vector_map=dict(vector_results)

             chunks=[]
             for rank , (idx , hybrid_score) in enumerate(fused):
                  chunk = copy.copy(self.chunks[idx])
                  chunk.bm25_score=bm25_map.get(idx)
                  chunk.vector_score=vector_map.get(idx)
                  chunk.hybrid_score=hybrid_score
                  chunk.rank_before_rerank=rank
                  chunks.append(chunk)

             return {
                  "query_id": str(uuid.uuid4()),
                  "query": query,
                  "chunks": chunks,
                  "retrieval_ms": retrieval_ms,
                  "timestamp": datetime.now(timezone.utc).isoformat(), 
             }





# if __name__ == "__main__":
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

#     result = retriever.retrieve("how does keyword search work?")

#     print(f"\nQuery: {result['query']}")
#     print(f"Retrieval time: {result['retrieval_ms']:.1f}ms\n")
#     for chunk in result["chunks"]:
#         print(f"  [{chunk.rank_before_rerank}] "
#         f"bm25={chunk.bm25_score or 0.0:.3f} "
#         f"vec={chunk.vector_score or 0.0:.3f} "
#         f"hybrid={chunk.hybrid_score:.4f}")
#         print(f"       {chunk.text[:60]}...")