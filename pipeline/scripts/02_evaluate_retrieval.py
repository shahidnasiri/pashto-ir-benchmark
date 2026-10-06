"""
Exact brute-force cosine retrieval for ONE model's saved embeddings
(Section 4.2). Computes Recall@{1,5,10,20}, MRR, nDCG@10 (Table 5),
and saves a per-query rank file used by every downstream script
(significance tests, stratification, error analysis).

USAGE (run once per model, after 01_encode_embeddings.py has produced
its .npy files - this script itself needs no GPU, so you can run it
in ANY of the 3 envs, or a plain base env with numpy/pandas):
    python scripts/02_evaluate_retrieval.py --model bge-m3

Run for all 10 models, then run 03/04/05/06 which read across all of
them at once.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from config import MODEL_REGISTRY, DISPLAY_NAMES, QUERIES_PATH, EMBED_DIR, RESULTS_DIR


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_ranks(doc_embs, query_embs, doc_ids, query_ids, targets):
    """Returns a DataFrame: query_id, target_document, rank (1-indexed)."""
    doc_id_to_idx = {d: i for i, d in enumerate(doc_ids)}
    # embeddings are already L2-normalized at encode time -> dot product = cosine
    sims = query_embs @ doc_embs.T  # (n_queries, n_docs)

    ranks = []
    for qi, qid in enumerate(query_ids):
        target_doc = targets[qid]
        target_idx = doc_id_to_idx[target_doc]
        scores = sims[qi]
        # rank = 1 + number of documents strictly more similar than the target
        rank = int((scores > scores[target_idx]).sum()) + 1
        ranks.append({"query_id": qid, "target_document": target_doc, "rank": rank})
    return pd.DataFrame(ranks)


def effectiveness_metrics(rank_df):
    ranks = rank_df["rank"].values
    n = len(ranks)
    metrics = {}
    for k in (1, 5, 10, 20):
        metrics[f"Recall@{k}"] = float((ranks <= k).mean())
    metrics["MRR"] = float((1.0 / ranks).mean())
    ndcg = np.where(ranks <= 10, 1.0 / np.log2(ranks + 1), 0.0)
    metrics["nDCG@10"] = float(ndcg.mean())
    return metrics


def append_table5_row(model_key, metrics):
    path = os.path.join(RESULTS_DIR, "table5_overall_effectiveness.csv")
    row = {"Model": DISPLAY_NAMES[model_key], **{k: round(v, 4) for k, v in metrics.items()}}
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df[df["Model"] != row["Model"]]  # overwrite if re-run
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df = df.sort_values("MRR", ascending=False).reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"Updated {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    args = parser.parse_args()
    key = args.model

    doc_embs = np.load(os.path.join(EMBED_DIR, f"{key}_docs.npy"))
    query_embs = np.load(os.path.join(EMBED_DIR, f"{key}_queries.npy"))
    with open(os.path.join(EMBED_DIR, f"{key}_doc_ids.json")) as f:
        doc_ids = json.load(f)
    with open(os.path.join(EMBED_DIR, f"{key}_query_ids.json")) as f:
        query_ids = json.load(f)

    queries = load_jsonl(QUERIES_PATH)
    targets = {q["query_id"]: q["target_document"] for q in queries}

    rank_df = compute_ranks(doc_embs, query_embs, doc_ids, query_ids, targets)

    # attach query metadata needed by 04_stratified_analysis.py
    meta = pd.DataFrame(queries)[["query_id", "difficulty", "query_type", "model"]]
    meta = meta.rename(columns={"model": "generating_model"})
    rank_df = rank_df.merge(meta, on="query_id", how="left")

    out_path = os.path.join(RESULTS_DIR, f"ranks_{key}.csv")
    rank_df.to_csv(out_path, index=False)
    print(f"Saved per-query ranks -> {out_path}")

    metrics = effectiveness_metrics(rank_df)
    append_table5_row(key, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
