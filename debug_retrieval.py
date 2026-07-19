# debug_retrieval.py

import sys
sys.path.insert(0, ".")

from src.document_loader import load_documents
from src.engine.retriever import HybridRetriever
from src.engine.reranker  import Reranker

texts, sources = load_documents("data/")

retriever = HybridRetriever()
retriever.index(texts, sources)
reranker  = Reranker()

queries = [
    "What does BERT stand for?",
    "What does RAG combine to generate answers?",
]

for query in queries:
    print(f"\nQuery: {query}")
    print("-" * 60)
    result  = retriever.retrieve(query)
    chunks  = reranker.rerank(query, result["chunks"])
    for c in chunks:
        print(f"  [{c.rank_after_rerank}] rerank={c.rerank_score:.3f} source={c.source}")
        print(f"       {c.text[:120]}...")
        print()