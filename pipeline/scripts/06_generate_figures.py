"""
Generates all 9 figures referenced in the paper, reading only from the
CSV/JSON files written by scripts 02-05. Run this last, after every
other script has been run for all 10 models.

Output: figures/figure1_... .png through figure9_... .png (300 DPI,
ready to paste into the Word doc placeholders).
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import re
from matplotlib import font_manager
import arabic_reshaper
from bidi.algorithm import get_display

ARABIC_RE = re.compile(r'[\u0600-\u06FF\u0750-\u077F]')
# Adjust this path if Noto Sans Arabic installed somewhere else on your system:
PASHTO_FONT = font_manager.FontProperties(
    fname=r"C:\Users\shahi\AppData\Local\Microsoft\Windows\Fonts\Bahij Zar-Regular.ttf")

def fix_rtl(text):
    text = str(text)
    if not ARABIC_RE.search(text):
        return text
    return get_display(arabic_reshaper.reshape(text))

from config import DOCS_PATH, QUERIES_PATH, RESULTS_DIR, FIGURES_DIR, MODEL_REGISTRY

plt.rcParams.update({
    "font.size": 11, "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#444444", "axes.labelcolor": "#222222",
})


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def savefig(name):
    path = os.path.join(FIGURES_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ---------------- Figure 1: word-length distributions ----------------
def figure1():
    docs = load_jsonl(DOCS_PATH)
    queries = load_jsonl(QUERIES_PATH)
    doc_lens = [len(d["document_text"].split()) for d in docs]
    query_lens = [len(q["query_text"].split()) for q in queries]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(doc_lens, bins=30, color="#2E86AB", edgecolor="white")
    axes[0].set_title("Document Length (words)")
    axes[0].set_xlabel("Word count"); axes[0].set_ylabel("Documents")
    axes[1].hist(query_lens, bins=20, color="#F18F01", edgecolor="white")
    axes[1].set_title("Query Length (words)")
    axes[1].set_xlabel("Word count"); axes[1].set_ylabel("Queries")
    savefig("figure1_word_length_distributions.png")


# ---------------- Figure 2: difficulty / query type distribution ----------------
def figure2():
    queries = load_jsonl(QUERIES_PATH)
    diff_counts = pd.Series([q["difficulty"] for q in queries]).value_counts()
    type_counts = pd.Series([q["query_type"] for q in queries]).value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(diff_counts.index, diff_counts.values, color="#3FA34D")
    axes[0].set_title("Query Difficulty Distribution")
    axes[1].bar(type_counts.index, type_counts.values, color="#2E86AB")
    axes[1].set_title("Query Type Distribution")
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")
    savefig("figure2_difficulty_type_distribution.png")


# ---------------- Figure 3: generator confusion matrices ----------------
def figure3():
    doc_path = os.path.join(RESULTS_DIR, "generator_confusion_matrix_documents.csv")
    query_path = os.path.join(RESULTS_DIR, "generator_confusion_matrix_queries.csv")
    if not (os.path.exists(doc_path) and os.path.exists(query_path)):
        print("Skipping Figure 3 - run script 05 first."); return
    doc_cm = pd.read_csv(doc_path, index_col=0)
    query_cm = pd.read_csv(query_path, index_col=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(doc_cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
    axes[0].set_title("Document-Level Classification"); axes[0].set_ylabel("True Generator")
    sns.heatmap(query_cm, annot=True, fmt="d", cmap="Oranges", ax=axes[1], cbar=False)
    axes[1].set_title("Query-Level Classification")
    for ax in axes:
        ax.set_xlabel("Predicted Generator")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    savefig("figure3_generator_confusion_matrices.png")


# ---------------- Figure 4: overall retrieval effectiveness ----------------
def figure4():
    path = os.path.join(RESULTS_DIR, "table5_overall_effectiveness.csv")
    if not os.path.exists(path):
        print("Skipping Figure 4 - run script 02 for all models first."); return
    df = pd.read_csv(path).sort_values("MRR", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(df["Model"], df["MRR"], color="#2E86AB")
    ax.set_xlabel("Mean Reciprocal Rank (MRR)")
    ax.set_title("Retrieval Effectiveness Across Embedding Models")
    ax.set_xlim(0, 1)
    for i, v in enumerate(df["MRR"]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    savefig("figure4_overall_effectiveness.png")


# ---------------- Figure 5: pairwise significance matrix ----------------
def figure5():
    path = os.path.join(RESULTS_DIR, "pairwise_pvalue_matrix_mrr.csv")
    if not os.path.exists(path):
        print("Skipping Figure 5 - run script 03 first."); return
    mat = pd.read_csv(path, index_col=0)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(mat, annot=True, fmt=".3f", cmap="RdYlGn_r", vmin=0, vmax=0.1,
                ax=ax, cbar_kws={"label": "Holm-adjusted p-value"})
    ax.set_title("Pairwise Statistical Significance (MRR, Holm-corrected)")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    savefig("figure5_pairwise_significance_matrix.png")


# ---------------- Figure 6: accuracy vs latency vs memory trade-off ----------------
def figure6():
    eff_path = os.path.join(RESULTS_DIR, "table7_efficiency.csv")
    acc_path = os.path.join(RESULTS_DIR, "table5_overall_effectiveness.csv")
    if not (os.path.exists(eff_path) and os.path.exists(acc_path)):
        print("Skipping Figure 6 - run scripts 01 and 02 for all models first."); return
    eff = pd.read_csv(eff_path)
    acc = pd.read_csv(acc_path)
    merged = eff.merge(acc, on="Model")

    fig, ax = plt.subplots(figsize=(9, 7))
    sizes = merged["Peak VRAM (GB)"] * 40
    scatter = ax.scatter(merged["Query Latency (ms)"], merged["MRR"], s=sizes,
                          c=merged["Peak VRAM (GB)"], cmap="viridis", alpha=0.8, edgecolor="black")
    for _, row in merged.iterrows():
        ax.annotate(row["Model"], (row["Query Latency (ms)"], row["MRR"]),
                    fontsize=8, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Query Latency (ms)"); ax.set_ylabel("MRR")
    ax.set_title("Accuracy vs. Latency (bubble size/color = Peak VRAM)")
    fig.colorbar(scatter, label="Peak VRAM (GB)")
    savefig("figure6_accuracy_latency_tradeoff.png")


# ---------------- Figure 7: MRR by category, all models ----------------
def figure7():
    path = os.path.join(RESULTS_DIR, "table10_category_similarity_vs_mrr.csv")
    if not os.path.exists(path):
        print("Skipping Figure 7 - run script 04 first."); return
    df = pd.read_csv(path).sort_values("Best-Model MRR")
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [fix_rtl(c) for c in df["Category"]]
    ax.barh(labels, df["Best-Model MRR"], color="#3FA34D")
    ax.set_yticklabels(labels, fontproperties=PASHTO_FONT, fontsize=10)
    ax.set_xlabel("Best-Model MRR"); ax.set_xlim(0, 1)
    ax.set_title("Retrieval Effectiveness Across Target Document Categories")
    savefig("figure7_mrr_by_category.png")


# ---------------- Figure 8: intra-category similarity vs MRR correlation ----------------
def figure8():
    path = os.path.join(RESULTS_DIR, "table10_category_similarity_vs_mrr.csv")
    if not os.path.exists(path):
        print("Skipping Figure 8 - run script 04 first."); return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(df["Intra-Category Similarity"], df["Best-Model MRR"], s=90, color="#D64545")
    for _, row in df.iterrows():
        ax.annotate(fix_rtl(row["Category"]), (row["Intra-Category Similarity"], row["Best-Model MRR"]),
                    fontsize=8, xytext=(5, 5), textcoords="offset points", fontproperties=PASHTO_FONT)
    corr = df["Intra-Category Similarity"].corr(df["Best-Model MRR"])
    ax.set_xlabel("Intra-Category TF-IDF Cosine Similarity")
    ax.set_ylabel("Best-Model MRR")
    ax.set_title(f"Category Homogeneity vs. Retrieval Effectiveness (r={corr:.3f})")
    savefig("figure8_similarity_vs_mrr_correlation.png")


# ---------------- Figure 9: cross-generator robustness ----------------
def figure9():
    path = os.path.join(RESULTS_DIR, "generator_robustness_gap.csv")
    if not os.path.exists(path):
        print("Skipping Figure 9 - run script 04 first."); return
    df = pd.read_csv(path).sort_values("Robustness Gap")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(df["Model"], df["Robustness Gap"], color="#F18F01")
    ax.set_xlabel("Robustness Gap (Best Generator MRR − Worst Generator MRR)")
    ax.set_title("Cross-Generator Robustness of Embedding Models")
    savefig("figure9_cross_generator_robustness.png")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure6()
    figure7()
    figure8()
    figure9()
    print("\nAll available figures written to figures/")
