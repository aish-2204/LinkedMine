# =============================================================================
# STEP 7 — DATA MINING: ASSOCIATION RULE MINING
# Rubric: 8.3 Association Rule Mining
# Task: Discover which skills frequently appear together in job postings
# Method: Apriori algorithm (mlxtend library)
# Output: Rules ranked by Support, Confidence, Lift
# =============================================================================
# Market basket analogy:
#   Transaction = one job posting's skill list
#   Item        = one skill (e.g. "Python", "SQL", "AWS")
#   Rule        = "If Python AND SQL → Machine Learning"
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings("ignore")

PROCESSED   = "data/processed"
FIGURES_DIR = "outputs/figures"


def load_skills():
    path = os.path.join(PROCESSED, "job_skills_clean.csv")
    if not os.path.exists(path):
        print("  job_skills_clean.csv not found — generating sample")
        return _sample_skills()
    return pd.read_csv(path)


def _sample_skills():
    """Sample skill data if Kaggle file unavailable."""
    np.random.seed(42)
    pool = ["Python","SQL","Machine Learning","Excel","Communication",
            "Leadership","Java","AWS","Tableau","R","TensorFlow",
            "Data Analysis","Project Management","Azure","Power BI",
            "Spark","NLP","Deep Learning","Git","Kubernetes"]
    rows = []
    for job_id in range(1, 2001):
        for skill in np.random.choice(pool, np.random.randint(2, 7), replace=False):
            rows.append({"job_id": job_id, "skill_abr": skill})
    return pd.DataFrame(rows)


def build_transactions(skills_df):
    """Convert skill records into list of transactions (one list per job)."""
    skill_col = "skill_abr" if "skill_abr" in skills_df.columns else skills_df.columns[-1]
    job_col   = "job_id"    if "job_id"    in skills_df.columns else skills_df.columns[0]

    # Keep only top 30 skills to ensure meaningful support
    top_skills = skills_df[skill_col].value_counts().head(30).index.tolist()
    filtered   = skills_df[skills_df[skill_col].isin(top_skills)]

    transactions = filtered.groupby(job_col)[skill_col].apply(list).tolist()
    print(f"  Transactions (job postings): {len(transactions):,}")
    print(f"  Unique items (skills):       {len(top_skills)}")
    return transactions, top_skills


def run_apriori(transactions, min_support=0.05, min_confidence=0.3):
    """Run Apriori and generate association rules."""
    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    df_enc   = pd.DataFrame(te_array, columns=te.columns_)

    # Frequent itemsets
    frequent_itemsets = apriori(df_enc, min_support=min_support, use_colnames=True)
    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)
    print(f"\n  Frequent itemsets found: {len(frequent_itemsets):,}")
    print(f"  (min_support={min_support}, min_confidence={min_confidence})")

    if len(frequent_itemsets) == 0:
        print("  No itemsets found — try lowering min_support")
        return None, None

    # Association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

    print(f"\n  Association rules generated: {len(rules):,}")
    return frequent_itemsets, rules


def display_top_rules(rules, top_n=20):
    """Print top rules in a readable format."""
    if rules is None or len(rules) == 0:
        return

    print(f"\n  TOP {top_n} RULES (by Lift)")
    print(f"  {'Antecedent':<30} {'Consequent':<25} {'Support':>8} {'Confidence':>12} {'Lift':>8}")
    print("  " + "-" * 88)

    for _, row in rules.head(top_n).iterrows():
        ant = ", ".join(list(row["antecedents"]))
        con = ", ".join(list(row["consequents"]))
        print(f"  {ant:<30} → {con:<25} {row['support']:>8.3f} {row['confidence']:>12.3f} {row['lift']:>8.2f}")


def plot_scatter_rules(rules):
    """Support vs Confidence scatter, bubble size = lift."""
    if rules is None or len(rules) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(
        rules["support"], rules["confidence"],
        s=rules["lift"] * 30, c=rules["lift"],
        cmap="Blues", alpha=0.7, edgecolors="none"
    )
    plt.colorbar(sc, ax=ax, label="Lift")
    ax.set_xlabel("Support", fontsize=11)
    ax.set_ylabel("Confidence", fontsize=11)
    ax.set_title("Association rules — support vs confidence (bubble size = lift)",
                 fontsize=12, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "13_association_rules_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_top_rules_bar(rules, top_n=15):
    """Horizontal bar chart of top rules by lift."""
    if rules is None or len(rules) < 2:
        return

    top = rules.head(top_n).copy()
    top["rule_label"] = (
        top["antecedents"].apply(lambda x: ", ".join(list(x))) +
        " → " +
        top["consequents"].apply(lambda x: ", ".join(list(x)))
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top["rule_label"][::-1], top["lift"][::-1],
                   color="#0A66C2", edgecolor="none")
    ax.set_xlabel("Lift", fontsize=11)
    ax.set_title(f"Top {top_n} association rules by lift", fontsize=12, fontweight="bold")
    ax.axvline(1.0, color="#e63946", linestyle="--", linewidth=1, label="Lift = 1 (random)")
    ax.legend(fontsize=9)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(left=False)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height()/2,
                f"{w:.2f}", va="center", fontsize=8, color="#555")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "14_top_rules_lift.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_skill_cooccurrence_heatmap(transactions, top_skills, top_n=15):
    """Build a co-occurrence matrix for the top N skills."""
    skills_subset = top_skills[:top_n]
    matrix = pd.DataFrame(0, index=skills_subset, columns=skills_subset)

    for transaction in transactions:
        present = [s for s in transaction if s in skills_subset]
        for i, s1 in enumerate(present):
            for s2 in present[i+1:]:
                matrix.loc[s1, s2] += 1
                matrix.loc[s2, s1] += 1

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.eye(len(skills_subset), dtype=bool)
    sns.heatmap(matrix, mask=mask, cmap="Blues", ax=ax,
                linewidths=0.3, annot=True, fmt="d",
                annot_kws={"size": 8})
    ax.set_title(f"Skill co-occurrence matrix (top {top_n} skills)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "15_skill_cooccurrence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 7: ASSOCIATION RULE MINING — Skill co-occurrence")
    print("=" * 60)

    skills_df = load_skills()
    transactions, top_skills = build_transactions(skills_df)

    frequent_itemsets, rules = run_apriori(transactions, min_support=0.05, min_confidence=0.3)
    display_top_rules(rules)

    plot_scatter_rules(rules)
    plot_top_rules_bar(rules)
    plot_skill_cooccurrence_heatmap(transactions, top_skills)

    # Save rules to CSV
    if rules is not None:
        rules_out = rules.copy()
        rules_out["antecedents"] = rules_out["antecedents"].apply(lambda x: ", ".join(list(x)))
        rules_out["consequents"] = rules_out["consequents"].apply(lambda x: ", ".join(list(x)))
        rules_out.to_csv("outputs/association_rules.csv", index=False)
        print("\n  Rules saved → outputs/association_rules.csv")

    print("\nStep 7 complete. Proceed to: python 08_mining_regression.py")
