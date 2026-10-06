"""
Encode all documents and queries with ONE embedding model and save:
  embeddings/{model_key}_docs.npy
  embeddings/{model_key}_queries.npy
  embeddings/{model_key}_doc_ids.json
  embeddings/{model_key}_query_ids.json
Also appends a row to results/table7_efficiency.csv (Doc Encode Time,
Query Encode Time, Throughput, Query Latency, Peak VRAM) matching
Table 7 of the paper.

USAGE (run from project root, inside the correct conda env for this model):
    python scripts/01_encode_embeddings.py --model nomic-embed-text-v2-moe

Run once per model, in whichever of the 3 conda envs that model belongs to
(see config.MODEL_REGISTRY[model]["env"]).
"""
import argparse
import csv
import json
import os
import time
import numpy as np
import pandas as pd

import numpy as np
import torch
from tqdm import tqdm

from config import (
    MODEL_REGISTRY, DISPLAY_NAMES, DOCS_PATH, QUERIES_PATH,
    EMBED_DIR, RESULTS_DIR, BATCH_SIZE,
)


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_with_sentence_transformers(cfg, texts, is_query, batch_size):
    from sentence_transformers import SentenceTransformer

    kwargs = {"trust_remote_code": cfg.get("trust_remote_code", False)}
    model = SentenceTransformer(cfg["hf_id"], device="cuda", **kwargs)

    # Jina-style task-conditioned encoding
    if "task_query" in cfg:
        task = cfg["task_query"] if is_query else cfg["task_doc"]
        embs = model.encode(
            texts, batch_size=batch_size, show_progress_bar=True,
            normalize_embeddings=True, task=task,
        )
    else:
        prefix = cfg["query_prefix"] if is_query else cfg["doc_prefix"]
        prefixed = [prefix + t for t in texts] if prefix else texts
        embs = model.encode(
            prefixed, batch_size=batch_size, show_progress_bar=True,
            normalize_embeddings=True,
        )
    del model
    torch.cuda.empty_cache()
    return np.asarray(embs, dtype=np.float32)


def encode_with_meanpool(cfg, texts, is_query, batch_size):
    """For raw MLM checkpoints with no sentence-transformers pooling head
    (Pashto BERT baseline, Section 4.1)."""
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"])
    model = AutoModel.from_pretrained(cfg["hf_id"]).to("cuda").eval()

    all_embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                             max_length=512, return_tensors="pt").to("cuda")
            out = model(**enc)
            pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_embs.append(pooled.cpu().numpy())

    del model
    torch.cuda.empty_cache()
    return np.concatenate(all_embs, axis=0).astype(np.float32)


def encode_texts(cfg, texts, is_query, batch_size):
    if cfg["loader"] == "sentence-transformers":
        return encode_with_sentence_transformers(cfg, texts, is_query, batch_size)
    elif cfg["loader"] == "transformers-meanpool":
        return encode_with_meanpool(cfg, texts, is_query, batch_size)
    else:
        raise ValueError(f"Unknown loader: {cfg['loader']}")


def append_efficiency_row(model_key, dim, doc_time, query_time, n_docs, n_queries, peak_vram_gb):
    
    path = os.path.join(RESULTS_DIR, "table7_efficiency.csv")
    throughput = n_docs / doc_time if doc_time > 0 else float("nan")
    latency_ms = (query_time / n_queries) * 1000 if n_queries > 0 else float("nan")
    new_row = {
        "Model": DISPLAY_NAMES[model_key], "Dim": dim,
        "Doc Encode Time (s)": round(doc_time, 2), "Query Encode Time (s)": round(query_time, 2),
        "Throughput (doc/s)": round(throughput, 1), "Query Latency (ms)": round(latency_ms, 1),
        "Peak VRAM (GB)": round(peak_vram_gb, 2),
    }
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df[df["Model"] != new_row["Model"]]  # drop any prior run for this model
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])
    df.to_csv(path, index=False)
    print(f"Updated efficiency row for {model_key} -> {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    cfg = MODEL_REGISTRY[args.model]
    print(f"Encoding with: {DISPLAY_NAMES[args.model]}  (hf_id={cfg['hf_id']}, env={cfg['env']})")

    docs = load_jsonl(DOCS_PATH)
    queries = load_jsonl(QUERIES_PATH)
    doc_ids = [d["document_id"] for d in docs]
    doc_texts = [d["document_text"] for d in docs]
    query_ids = [q["query_id"] for q in queries]
    query_texts = [q["query_text"] for q in queries]

    assert torch.cuda.is_available(), "CUDA not available in this environment!"
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    doc_embs = encode_texts(cfg, doc_texts, is_query=False, batch_size=args.batch_size)
    doc_time = time.time() - t0

    t1 = time.time()
    query_embs = encode_texts(cfg, query_texts, is_query=True, batch_size=args.batch_size)
    query_time = time.time() - t1

    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9

    np.save(os.path.join(EMBED_DIR, f"{args.model}_docs.npy"), doc_embs)
    np.save(os.path.join(EMBED_DIR, f"{args.model}_queries.npy"), query_embs)
    with open(os.path.join(EMBED_DIR, f"{args.model}_doc_ids.json"), "w") as f:
        json.dump(doc_ids, f)
    with open(os.path.join(EMBED_DIR, f"{args.model}_query_ids.json"), "w") as f:
        json.dump(query_ids, f)

    append_efficiency_row(args.model, cfg["dim"], doc_time, query_time,
                           len(doc_texts), len(query_texts), peak_vram_gb)

    print(f"Done. doc_embs shape={doc_embs.shape}, query_embs shape={query_embs.shape}")
    print(f"Doc encode: {doc_time:.1f}s | Query encode: {query_time:.1f}s | Peak VRAM: {peak_vram_gb:.2f} GB")


if __name__ == "__main__":
    main()
