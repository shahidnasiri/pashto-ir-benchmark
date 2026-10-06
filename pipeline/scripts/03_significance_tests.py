"""
Statistical significance testing across all 10 models (Section 4.5 / 5.2).
Reads results/ranks_{model}.csv for every model (produced by script 02),
and produces:
  results/table6_significance_tiers.csv
  results/pairwise_pvalue_matrix_mrr.csv       (Figure 5 source data)
  results/pairwise_pvalue_matrix_ndcg10.csv    (robustness check, Section 5.2)
  results/bootstrap_cis.csv

Run this AFTER script 02 has been run for all 10 models.
No GPU needed - run in any env with scipy/pandas installed.
"""
import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from config import MODEL_REGISTRY, DISPLAY_NAMES, RESULTS_DIR, RANDOM_SEED

N_BOOTSTRAP = 10000
ALPHA = 0.05


def load_all_ranks():
    per_model = {}
    for key in MODEL_REGISTRY:
        path = os.path.join(RESULTS_DIR, f"ranks_{key}.csv")
        if not os.path.exists(path):
            print(f"WARNING: missing {path}, skipping {key}")
            continue
        df = pd.read_csv(path).sort_values("query_id").reset_index(drop=True)
        per_model[key] = df
    return per_model


def holm_correction(pvalues):
    """Standard Holm step-down procedure. Returns adjusted p-values
    in the ORIGINAL order of `pvalues`."""
    n = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * pvalues[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def bootstrap_ci_mean_diff(a, b, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = a - b
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_means[i] = diffs[idx].mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(lo), float(hi)


def pairwise_tests(per_model, metric_col):
    """metric_col: 'rr' (reciprocal rank -> MRR) or 'ndcg' (nDCG@10 per query)."""
    keys = list(per_model.keys())
    n = len(keys)
    raw_p = {}
    mean_diff = {}
    pairs = list(itertools.combinations(range(n), 2))

    pvals_list = []
    for i, j in pairs:
        a = per_model[keys[i]][metric_col].values
        b = per_model[keys[j]][metric_col].values
        try:
            stat, p = wilcoxon(a, b)
        except ValueError:
            p = 1.0  # identical distributions (e.g. all-zero diffs)
        raw_p[(i, j)] = p
        mean_diff[(i, j)] = float((a - b).mean())
        pvals_list.append(p)

    adj_p_array = holm_correction(np.array(pvals_list))
    adj_p = {pair: adj_p_array[idx] for idx, pair in enumerate(pairs)}

    arr = np.full((n, n), np.nan)
    for (i, j), p in adj_p.items():
        arr[i, j] = p
        arr[j, i] = p
    np.fill_diagonal(arr, 1.0)
    matrix = pd.DataFrame(arr, index=[DISPLAY_NAMES[k] for k in keys],
                           columns=[DISPLAY_NAMES[k] for k in keys])
    return matrix, adj_p, mean_diff, keys


def build_tiers(keys, adj_p, mean_mrr):
    """Simplified transitive tiering: walk the ranking; start a new tier
    whenever the gap to the previous model is statistically significant."""
    ranking = sorted(keys, key=lambda k: -mean_mrr[k])
    tiers = {}
    tier_num = 1
    tiers[ranking[0]] = tier_num
    for prev, curr in zip(ranking, ranking[1:]):
        i, j = keys.index(prev), keys.index(curr)
        pair = (min(i, j), max(i, j))
        p = adj_p.get(pair, 1.0)
        if p < ALPHA:
            tier_num += 1
        tiers[curr] = tier_num
    return tiers, ranking


def main():
    per_model = load_all_ranks()
    if len(per_model) < 2:
        print("Need at least 2 models' rank files to run significance tests.")
        return

    for df in per_model.values():
        df["rr"] = 1.0 / df["rank"]
        df["ndcg"] = np.where(df["rank"] <= 10, 1.0 / np.log2(df["rank"] + 1), 0.0)

    mean_mrr = {k: df["rr"].mean() for k, df in per_model.items()}

    # --- MRR-based significance (primary) ---
    matrix_mrr, adj_p_mrr, mean_diff_mrr, keys = pairwise_tests(per_model, "rr")
    matrix_mrr.to_csv(os.path.join(RESULTS_DIR, "pairwise_pvalue_matrix_mrr.csv"))

    # --- nDCG@10-based significance (robustness check, Section 5.2) ---
    matrix_ndcg, adj_p_ndcg, _, _ = pairwise_tests(per_model, "ndcg")
    matrix_ndcg.to_csv(os.path.join(RESULTS_DIR, "pairwise_pvalue_matrix_ndcg10.csv"))

    # Agreement rate between the two metrics' significance conclusions
    sig_mrr = {pair: (p < ALPHA) for pair, p in adj_p_mrr.items()}
    sig_ndcg = {pair: (p < ALPHA) for pair, p in adj_p_ndcg.items()}
    agree = sum(sig_mrr[p] == sig_ndcg[p] for p in sig_mrr) / len(sig_mrr)
    print(f"MRR vs nDCG@10 significance agreement: {agree*100:.1f}%")

    # --- Tiering ---
    tiers, ranking = build_tiers(keys, adj_p_mrr, mean_mrr)
    tier_rows = [{"Tier": tiers[k], "Model": DISPLAY_NAMES[k],
                  "Mean MRR": round(mean_mrr[k], 4)} for k in ranking]
    tier_df = pd.DataFrame(tier_rows)
    tier_df.to_csv(os.path.join(RESULTS_DIR, "table6_significance_tiers.csv"), index=False)
    print(tier_df.to_string(index=False))

    n_sig = sum(sig_mrr.values())
    n_total = len(sig_mrr)
    print(f"\n{n_sig}/{n_total} pairwise comparisons statistically significant (Holm, alpha={ALPHA})")

    # --- Bootstrap CI for every adjacent pair in the ranking (headline comparisons) ---
    ci_rows = []
    for prev, curr in zip(ranking, ranking[1:]):
        a = per_model[prev]["rr"].values
        b = per_model[curr]["rr"].values
        lo, hi = bootstrap_ci_mean_diff(a, b)
        i, j = keys.index(prev), keys.index(curr)
        p = adj_p_mrr.get((min(i, j), max(i, j)), float("nan"))
        ci_rows.append({
            "Model A": DISPLAY_NAMES[prev], "Model B": DISPLAY_NAMES[curr],
            "Mean MRR diff": round(float((a - b).mean()), 4),
            "95% CI low": round(lo, 4), "95% CI high": round(hi, 4),
            "Holm-adjusted p": round(p, 4),
        })
    pd.DataFrame(ci_rows).to_csv(os.path.join(RESULTS_DIR, "bootstrap_cis.csv"), index=False)
    print(f"\nSaved bootstrap CIs -> results/bootstrap_cis.csv")
    print(f"Saved significance matrices -> results/pairwise_pvalue_matrix_{{mrr,ndcg10}}.csv")


if __name__ == "__main__":
    main()
