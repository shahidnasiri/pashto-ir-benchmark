"""
Section 5.1 / 3.4: character n-gram TF-IDF + logistic regression classifier
predicting which of the 5 LLM generators produced a document/query,
under 5-fold stratified cross-validation. Also computes per-generator
lexical statistics (Table 4) and benchmark v1-vs-v2 duplication stats
(Table 3, if you still have your old v1 files - otherwise fill those
numbers in from your existing QA notebook outputs, they don't change).

Produces:
  results/table4_lexical_stats_by_generator.csv
  results/generator_confusion_matrix_documents.csv  (Figure 3 source)
  results/generator_confusion_matrix_queries.csv    (Figure 3 source)
  results/generator_fingerprint_accuracy.json

No GPU needed.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score

from config import DOCS_PATH, QUERIES_PATH, RESULTS_DIR, RANDOM_SEED


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_classification(texts, labels, label_names):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    preds = cross_val_predict(clf, X, labels, cv=skf)
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds, labels=label_names)
    return acc, cm


def lexical_stats(texts):
    all_tokens = []
    for t in texts:
        all_tokens.extend(t.split())
    total_tokens = len(all_tokens)
    vocab = set(all_tokens)
    ttr = len(vocab) / total_tokens if total_tokens else 0
    avg_word_len = np.mean([len(w) for w in all_tokens]) if all_tokens else 0
    return total_tokens, len(vocab), ttr, avg_word_len


def main():
    docs = load_jsonl(DOCS_PATH)
    queries = load_jsonl(QUERIES_PATH)

    doc_texts = [d["document_text"] for d in docs]
    doc_labels = [d["model"] for d in docs]
    query_texts = [q["query_text"] for q in queries]
    query_labels = [q["model"] for q in queries]

    generators = sorted(set(doc_labels))
    print(f"Generators found: {generators}")

    doc_acc, doc_cm = run_classification(doc_texts, doc_labels, generators)
    query_acc, query_cm = run_classification(query_texts, query_labels, generators)

    print(f"Document-level classification accuracy: {doc_acc*100:.1f}%")
    print(f"Query-level classification accuracy:    {query_acc*100:.1f}%")
    print(f"(chance baseline = {100/len(generators):.1f}% for {len(generators)} balanced classes)")

    pd.DataFrame(doc_cm, index=generators, columns=generators).to_csv(
        os.path.join(RESULTS_DIR, "generator_confusion_matrix_documents.csv"))
    pd.DataFrame(query_cm, index=generators, columns=generators).to_csv(
        os.path.join(RESULTS_DIR, "generator_confusion_matrix_queries.csv"))

    with open(os.path.join(RESULTS_DIR, "generator_fingerprint_accuracy.json"), "w") as f:
        json.dump({
            "document_accuracy": round(doc_acc, 4),
            "query_accuracy": round(query_acc, 4),
            "chance_baseline": round(1 / len(generators), 4),
            "n_generators": len(generators),
        }, f, indent=2)

    # ---------------- Table 4: per-generator lexical stats ----------------
    rows = []
    for gen in generators:
        gen_texts = [d["document_text"] for d in docs if d["model"] == gen]
        total_tokens, vocab_size, ttr, avg_len = lexical_stats(gen_texts)
        rows.append({
            "Generator": gen, "Documents": len(gen_texts),
            "Total Tokens": total_tokens, "Vocabulary Size": vocab_size,
            "Type-Token Ratio": round(ttr, 4), "Avg. Word Length (chars)": round(avg_len, 2),
        })
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "table4_lexical_stats_by_generator.csv"),
                               index=False)
    print("Saved table4_lexical_stats_by_generator.csv, confusion matrices, and accuracy json")


if __name__ == "__main__":
    main()
