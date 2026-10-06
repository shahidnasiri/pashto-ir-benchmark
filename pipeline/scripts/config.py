"""
Central configuration: the 10-model registry (matches Table 1 of the paper),
file paths, and shared constants. Every other script imports from here.
"""
import os

# ---------------------------------------------------------------
# Paths (relative to project root; scripts assume they're run from
# the project root, e.g. `python scripts/01_encode_embeddings.py`)
# ---------------------------------------------------------------
DATA_DIR = "PASHTO_IR_BENCHMARK_V2_FINAL"
EMBED_DIR = "embeddings"
RESULTS_DIR = "results"
FIGURES_DIR = "figures"

DOCS_PATH = os.path.join(DATA_DIR, "pashto_ir_documents_v2_final.jsonl")
QUERIES_PATH = os.path.join(DATA_DIR, "pashto_ir_queries_v2_final.jsonl")

for d in (EMBED_DIR, RESULTS_DIR, FIGURES_DIR):
    os.makedirs(d, exist_ok=True)

RANDOM_SEED = 42

# ---------------------------------------------------------------
# Model registry - matches paper Table 1 exactly
#   env: which conda environment this model must be run in
#   loader: dispatch key used by 01_encode_embeddings.py
#   query_prefix / doc_prefix: text prepended before encoding
#   task_query / task_doc: Jina-style task-conditioned encoding
# ---------------------------------------------------------------
MODEL_REGISTRY = {
    "nomic-embed-text-v2-moe": {
        "hf_id": "nomic-ai/nomic-embed-text-v2-moe",
        "dim": 768,
        "env": "nomic",
        "loader": "sentence-transformers",
        "trust_remote_code": True,
        "query_prefix": "search_query: ",
        "doc_prefix": "search_document: ",
    },
    "gte-multilingual-base": {
        "hf_id": "Alibaba-NLP/gte-multilingual-base",
        "dim": 768,
        "env": "jina_gte",
        "loader": "sentence-transformers",
        "trust_remote_code": True,
        "query_prefix": "",
        "doc_prefix": "",
    },
    "jina-embeddings-v3": {
        "hf_id": "jinaai/jina-embeddings-v3",
        "dim": 1024,
        "env": "jina_gte",
        "loader": "sentence-transformers",
        "trust_remote_code": True,
        "task_query": "retrieval.query",
        "task_doc": "retrieval.passage",
    },
    "labse": {
        "hf_id": "sentence-transformers/LaBSE",
        "dim": 768,
        "env": "main",
        "loader": "sentence-transformers",
        "query_prefix": "",
        "doc_prefix": "",
    },
    "paraphrase-mpnet": {
        "hf_id": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "dim": 768,
        "env": "main",
        "loader": "sentence-transformers",
        "query_prefix": "",
        "doc_prefix": "",
    },
    "paraphrase-minilm": {
        "hf_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "dim": 384,
        "env": "main",
        "loader": "sentence-transformers",
        "query_prefix": "",
        "doc_prefix": "",
    },
    "multilingual-e5-large-instruct": {
        "hf_id": "intfloat/multilingual-e5-large-instruct",
        "dim": 1024,
        "env": "main",
        "loader": "sentence-transformers",
        "query_prefix": "Instruct: Given a question, retrieve relevant passages that answer the question\nQuery: ",
        "doc_prefix": "",
    },
    "bge-m3": {
        "hf_id": "BAAI/bge-m3",
        "dim": 1024,
        "env": "main",
        "loader": "sentence-transformers",
        "query_prefix": "",
        "doc_prefix": "",
    },
    "qwen3-embedding-0.6b": {
        "hf_id": "Qwen/Qwen3-Embedding-0.6B",
        "dim": 1024,
        "env": "main",
        "loader": "sentence-transformers",
        "query_prefix": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
        "doc_prefix": "",
    },
    "pashto-bert": {
        "hf_id": "zirak-ai/pashto-bert-v1",
        "dim": 768,
        "env": "main",
        "loader": "transformers-meanpool",
        "query_prefix": "",
        "doc_prefix": "",
    },
}

# Canonical display names used in all output tables (matches paper exactly)
DISPLAY_NAMES = {
    "nomic-embed-text-v2-moe": "Nomic-Embed-Text-v2-MoE",
    "bge-m3": "BGE-M3",
    "multilingual-e5-large-instruct": "Multilingual-E5-large-instruct",
    "jina-embeddings-v3": "Jina-Embeddings-v3",
    "gte-multilingual-base": "GTE-multilingual-base",
    "labse": "LaBSE",
    "qwen3-embedding-0.6b": "Qwen3-Embedding-0.6B",
    "paraphrase-mpnet": "Paraphrase-multilingual-mpnet-base-v2",
    "paraphrase-minilm": "Paraphrase-multilingual-MiniLM-L12-v2",
    "pashto-bert": "Pashto BERT (baseline)",
}

MODELS_BY_ENV = {
    "main": [k for k, v in MODEL_REGISTRY.items() if v["env"] == "main"],
    "jina_gte": [k for k, v in MODEL_REGISTRY.items() if v["env"] == "jina_gte"],
    "nomic": [k for k, v in MODEL_REGISTRY.items() if v["env"] == "nomic"],
}

BATCH_SIZE = 16          # lower to 8 if you hit OOM on Qwen3/Jina/BGE-M3
QUERY_DIFFICULTIES = ["easy", "medium", "hard"]
QUERY_TYPES = ["direct_information", "procedural", "requirement", "indirect"]
