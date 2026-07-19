# models
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"

# retrieval
TOP_K_RETRIEVE=50
TOP_K_RERANK=5

# rrf fusion
BM25_WEIGHT=0.3
VECTOR_WEIGHT=0.7
RRF_K=60

# chunking
CHUNK_SIZE=300
CHUNK_OVERLAP=75
