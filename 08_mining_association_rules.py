# =============================================================================
# STEP 8 — DATA MINING: ASSOCIATION RULE MINING
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
    """Labeled bubble chart: x=Support, y=Confidence, bubble=Lift, each point named."""
    if rules is None or len(rules) == 0:
        return

    df = rules.copy()
    df["ant"] = df["antecedents"].apply(lambda x: _full_name(list(x)))
    df["con"] = df["consequents"].apply(lambda x: _full_name(list(x)))

    # Pair colors so both directions of the same rule share a color
    pair_colors = ["#0A66C2", "#0A66C2", "#e63946", "#e63946", "#2a9d8f", "#2a9d8f"]
    colors = pair_colors[:len(df)]

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        df["support"], df["confidence"],
        s=df["lift"] * 120, c=colors,
        alpha=0.80, edgecolors="white", linewidths=1.5, zorder=4
    )

    # Label every bubble with the rule
    for i, (_, row) in enumerate(df.iterrows()):
        label = f"{row['ant']}\n→ {row['con']}"
        offsets = [(10, 8), (-10, -20), (10, 8), (-10, -20), (10, 8), (-10, -20)]
        ox, oy = offsets[i] if i < len(offsets) else (10, 8)
        ax.annotate(
            label,
            xy=(row["support"], row["confidence"]),
            xytext=(ox, oy), textcoords="offset points",
            fontsize=8, color="#222",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#ccc", alpha=0.9, linewidth=0.8),
            arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.8),
            zorder=5
        )

    # Legend explaining bubble size
    for lift_val, label in [(3.5, "Lift 3.5×"), (5.5, "Lift 5.5×")]:
        ax.scatter([], [], s=lift_val * 120, c="#888", alpha=0.6, label=label)
    ax.legend(title="Bubble size", fontsize=8.5, title_fontsize=8.5,
              loc="lower right", framealpha=0.9)

    ax.set_xlabel("Support  (fraction of all job postings containing both categories)", fontsize=10)
    ax.set_ylabel("Confidence  (when IF category appears, how often THEN also appears)", fontsize=10)
    ax.set_title(
        "Association Rules — Support vs Confidence\n"
        "Larger bubble = stronger Lift (more surprising co-occurrence)",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "13_association_rules_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


CATEGORY_LABELS = {
    'IT': 'Info Technology', 'SALE': 'Sales',   'MGMT': 'Management',
    'MNFC': 'Manufacturing', 'HCPR': 'Healthcare', 'BD': 'Business Dev',
    'ENG': 'Engineering',    'FIN': 'Finance',   'MRKT': 'Marketing',
    'ANLS': 'Analytics',     'ACCT': 'Accounting',
}


def _full_name(codes):
    return " + ".join(CATEGORY_LABELS.get(c.strip(), c.strip()) for c in codes)


def plot_top_rules_bar(rules, top_n=15):
    """3-panel metric chart: Support | Confidence | Lift — one bar per rule, all labeled."""
    if rules is None or len(rules) < 2:
        return

    top = rules.head(top_n).copy().reset_index(drop=True)
    top["ant_label"] = top["antecedents"].apply(lambda x: _full_name(list(x)))
    top["con_label"] = top["consequents"].apply(lambda x: _full_name(list(x)))
    top["rule_label"] = top.apply(
        lambda r: f"{r['ant_label']}  →  {r['con_label']}", axis=1
    )
    top = top.sort_values("lift", ascending=True).reset_index(drop=True)

    n = len(top)
    row_h = 0.75
    fig_h  = max(4.5, n * row_h + 2.5)

    fig, axes = plt.subplots(1, 3, figsize=(15, fig_h), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    metrics = [
        ("support",    "#2a9d8f", "Support",
         "% of ALL job postings\nthat contain BOTH categories",
         "{:.1%}"),
        ("confidence", "#0A66C2", "Confidence",
         "When IF category is present,\nhow often is THEN also present?",
         "{:.0%}"),
        ("lift",       "#e63946", "Lift",
         "How many times MORE likely\nthan random chance (>1 = meaningful)",
         "{:.2f}×"),
    ]

    y = list(range(n))

    for ax, (metric, color, title, subtitle, fmt) in zip(axes, metrics):
        vals = top[metric].values
        bars = ax.barh(y, vals, color=color, alpha=0.75, height=0.6, zorder=3)

        # Value labels at bar end
        for i, val in enumerate(vals):
            ax.text(val + vals.max() * 0.02, i, fmt.format(val),
                    va="center", ha="left", fontsize=9,
                    fontweight="bold", color=color)

        ax.set_xlim(0, vals.max() * 1.40)
        ax.set_title(f"{title}\n{subtitle}", fontsize=9.5,
                     fontweight="bold", color=color, pad=10)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(bottom=False, left=False)
        ax.set_xticks([])

        # Dashed baseline for Lift at 1.0
        if metric == "lift":
            ax.axvline(1.0, color="#bbb", linewidth=1.2,
                       linestyle="--", zorder=2)
            ax.text(1.0, -0.8, "random\nbaseline",
                    fontsize=7, color="#aaa", ha="center")

        # Zebra row background
        for i in y:
            ax.axhspan(i - 0.45, i + 0.45,
                       color="#f5f5f5" if i % 2 == 0 else "white",
                       zorder=1)

    # Rule labels on the left y-axis
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(top["rule_label"], fontsize=9.5)
    axes[0].tick_params(left=False)

    fig.suptitle(
        "Apriori Association Rules — Which Job Categories Appear Together?\n"
        "Ranked by Lift (strongest co-occurrence signal at top)",
        fontsize=12, fontweight="bold", y=1.01
    )

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "14_top_rules_lift.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.join(FIGURES_DIR, '14_top_rules_lift.png')}")


def plot_skill_cooccurrence_heatmap(transactions, top_skills, top_n=15):
    """Co-occurrence matrix with full human-readable category names."""
    skills_subset = top_skills[:top_n]
    readable      = [CATEGORY_LABELS.get(s, s) for s in skills_subset]
    matrix        = pd.DataFrame(0, index=readable, columns=readable)

    for transaction in transactions:
        present = [CATEGORY_LABELS.get(s, s) for s in transaction if s in skills_subset]
        for i, s1 in enumerate(present):
            for s2 in present[i+1:]:
                matrix.loc[s1, s2] += 1
                matrix.loc[s2, s1] += 1

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.eye(len(readable), dtype=bool)
    sns.heatmap(matrix, mask=mask, cmap="Blues", ax=ax,
                linewidths=0.3, annot=True, fmt=",d",
                annot_kws={"size": 7})
    ax.set_title(
        f"How often do job categories appear together? (top {top_n} categories)\n"
        "Each cell = number of job postings tagged with BOTH categories",
        fontsize=11, fontweight="bold", pad=12
    )
    ax.tick_params(axis="x", rotation=40, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "15_skill_cooccurrence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 8: ASSOCIATION RULE MINING — Skill co-occurrence")
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

    print("\nStep 8 complete. Proceed to: python 09_mining_regression.py")
