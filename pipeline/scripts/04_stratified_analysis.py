"""
Section 5.3 stratified analysis across all models: by difficulty (Table 8),
query type (Table 9), document category + intra-category similarity
(Table 10), generating model (Table 11, + robustness gap for Figure 9),
and the low-lexical-overlap query subset (Table 12).

Run AFTER scripts 02 (and ideally 03) for all 10 models.
No GPU needed.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import MODEL_REGISTRY, DISPLAY_NAMES, DOCS_PATH, QUERIES_PATH, RESULTS_DIR


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def char_ngram_tfidf(texts, ngram_range=(2, 4)):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range)
    return vec.fit_transform(texts), vec


def load_all_ranks():
    per_model = {}
    for key in MODEL_REGISTRY:
        path = os.path.join(RESULTS_DIR, f"ranks_{key}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["rr"] = 1.0 / df["rank"]
            per_model[key] = df
    return per_model


def pivot_mrr_by(per_model, group_col, categories=None):
    rows = []
    for key, df in per_model.items():
        row = {"Model": DISPLAY_NAMES[key]}
        groups = categories if categories else sorted(df[group_col].dropna().unique())
        for g in groups:
            sub = df[df[group_col] == g]
            row[str(g)] = round(sub["rr"].mean(), 4) if len(sub) else float("nan")
        rows.append(row)
    out = pd.DataFrame(rows)
    # order by overall MRR descending to match paper tables
    overall = {DISPLAY_NAMES[k]: df["rr"].mean() for k, df in per_model.items()}
    out["_sort"] = out["Model"].map(overall)
    out = out.sort_values("_sort", ascending=False).drop(columns="_sort").reset_index(drop=True)
    return out


def main():
    per_model = load_all_ranks()
    if not per_model:
        print("No ranks_*.csv files found. Run script 02 for all models first.")
        return

    docs = load_jsonl(DOCS_PATH)
    queries = load_jsonl(QUERIES_PATH)
    doc_by_id = {d["document_id"]: d for d in docs}

    # ---------------- Table 8: by difficulty ----------------
    from config import QUERY_DIFFICULTIES
    t8 = pivot_mrr_by(per_model, "difficulty", categories=["easy", "medium", "hard"])
    t8.columns = [c.capitalize() if c in ("easy", "medium", "hard") else c for c in t8.columns]
    t8.to_csv(os.path.join(RESULTS_DIR, "table8_mrr_by_difficulty.csv"), index=False)
    print("Saved table8_mrr_by_difficulty.csv")

    # ---------------- Table 9: by query type ----------------
    from config import QUERY_TYPES
    t9 = pivot_mrr_by(per_model, "query_type", categories=QUERY_TYPES)
    t9.to_csv(os.path.join(RESULTS_DIR, "table9_mrr_by_query_type.csv"), index=False)
    print("Saved table9_mrr_by_query_type.csv")

    # ---------------- Table 10: by document category + intra-category similarity ----------------
    # attach target document's category to each query's rank row
    doc_category = {d["document_id"]: d["category"] for d in docs}
    for df in per_model.values():
        df["category"] = df["target_document"].map(doc_category)

    categories = sorted(set(doc_category.values()))
    t10_rows = []
    for cat in categories:
        cat_docs = [d for d in docs if d["category"] == cat]
        texts = [d["document_text"] for d in cat_docs]
        if len(texts) >= 2:
            tfidf, _ = char_ngram_tfidf(texts)
            sim = cosine_similarity(tfidf)
            np.fill_diagonal(sim, np.nan)
            intra_sim = np.nanmean(sim)
        else:
            intra_sim = float("nan")

        best_mrr = -1
        for key, df in per_model.items():
            sub = df[df["category"] == cat]
            if len(sub):
                m = sub["rr"].mean()
                best_mrr = max(best_mrr, m)
        t10_rows.append({
            "Category": cat, "N Docs": len(cat_docs),
            "Intra-Category Similarity": round(intra_sim, 4),
            "Best-Model MRR": round(best_mrr, 4),
        })
    t10 = pd.DataFrame(t10_rows).sort_values("Best-Model MRR").reset_index(drop=True)
    t10.to_csv(os.path.join(RESULTS_DIR, "table10_category_similarity_vs_mrr.csv"), index=False)
    corr = t10["Intra-Category Similarity"].corr(t10["Best-Model MRR"])
    print(f"Saved table10 (Intra-category similarity vs MRR correlation r={corr:.3f})")

    # ---------------- Table 11: by generating model + robustness gap ----------------
    t11 = pivot_mrr_by(per_model, "generating_model")
    t11.to_csv(os.path.join(RESULTS_DIR, "table11_mrr_by_generating_model.csv"), index=False)

    gen_cols = [c for c in t11.columns if c != "Model"]
    gap_rows = []
    for _, row in t11.iterrows():
        vals = row[gen_cols].astype(float)
        gap_rows.append({"Model": row["Model"], "Best Generator MRR": round(vals.max(), 4),
                          "Worst Generator MRR": round(vals.min(), 4),
                          "Robustness Gap": round(vals.max() - vals.min(), 4)})
    gap_df = pd.DataFrame(gap_rows).sort_values("Robustness Gap")
    gap_df.to_csv(os.path.join(RESULTS_DIR, "generator_robustness_gap.csv"), index=False)
    print("Saved table11_mrr_by_generating_model.csv and generator_robustness_gap.csv")

    # ---------------- Table 12: low-lexical-overlap subset ----------------
    # Recompute the 5th-percentile low-overlap query set (Section 3.5):
    # char n-gram TF-IDF cosine similarity between each query and its target doc.
    all_texts = [q["query_text"] for q in queries] + [d["document_text"] for d in docs]
    tfidf_all, vec = char_ngram_tfidf(all_texts)
    n_q = len(queries)
    query_vecs = tfidf_all[:n_q]
    doc_vecs = tfidf_all[n_q:]
    doc_id_to_row = {d["document_id"]: i for i, d in enumerate(docs)}

    overlap_scores = []
    for qi, q in enumerate(queries):
        target_row = doc_id_to_row[q["target_document"]]
        sim = cosine_similarity(query_vecs[qi], doc_vecs[target_row])[0, 0]
        overlap_scores.append(sim)
    overlap_scores = np.array(overlap_scores)
    threshold = np.percentile(overlap_scores, 5)
    low_overlap_ids = [queries[i]["query_id"] for i in range(n_q) if overlap_scores[i] <= threshold]
    print(f"Low-overlap threshold (5th pct): {threshold:.4f}  -> {len(low_overlap_ids)} queries flagged")

    with open(os.path.join(RESULTS_DIR, "low_overlap_query_ids.json"), "w") as f:
        json.dump(low_overlap_ids, f)

    t12_rows = []
    for key, df in per_model.items():
        low_mask = df["query_id"].isin(low_overlap_ids)
        normal_mrr = df.loc[~low_mask, "rr"].mean()
        low_mrr = df.loc[low_mask, "rr"].mean()
        rel_drop = (normal_mrr - low_mrr) / normal_mrr * 100 if normal_mrr > 0 else float("nan")
        t12_rows.append({
            "Model": DISPLAY_NAMES[key],
            f"Normal-Overlap MRR (n={(~low_mask).sum()})": round(normal_mrr, 4),
            f"Low-Overlap MRR (n={low_mask.sum()})": round(low_mrr, 4),
            "Relative Drop": f"{rel_drop:.1f}%",
        })
    t12 = pd.DataFrame(t12_rows)
    # sort by relative drop ascending to match paper ordering
    t12["_drop_num"] = t12["Relative Drop"].str.rstrip("%").astype(float)
    t12 = t12.sort_values("_drop_num").drop(columns="_drop_num").reset_index(drop=True)
    t12.to_csv(os.path.join(RESULTS_DIR, "table12_low_overlap_subset.csv"), index=False)
    print("Saved table12_low_overlap_subset.csv")

    print("\nAll stratified tables written to results/")


if __name__ == "__main__":
    main()
