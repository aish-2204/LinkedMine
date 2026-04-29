# =============================================================================
# STEP 4 — EXPLORATORY DATA ANALYSIS (EDA)
# Rubric: Understand the dataset before modelling
# Output: Figure 09 — Salary distribution + Job category tags per posting
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

PROCESSED   = "data/processed"
FIGURES_DIR = "outputs/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

PALETTE = "#0A66C2"


def load():
    path = os.path.join(PROCESSED, "jobs_preprocessed.csv")
    if not os.path.exists(path):
        path = os.path.join(PROCESSED, "jobs_merged.csv")
    return pd.read_csv(path, low_memory=False)


# =============================================================================
# DATASET OVERVIEW (printed to console)
# =============================================================================
def print_overview(df):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    null_total = df.isnull().sum().sum()
    null_pct   = null_total / df.size * 100

    print(f"\n  Shape         : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Memory        : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"  Numeric cols  : {len(num_cols)}")
    print(f"  Categorical   : {len(cat_cols)}")
    print(f"  Total nulls   : {null_total:,}  ({null_pct:.1f}% of all cells)")

    print(f"\n  Columns with missing values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    for col, n in missing.items():
        print(f"    {col:<35} {n:>7,}  ({n/len(df)*100:.1f}%)")


# =============================================================================
# KEY VARIABLE DISTRIBUTIONS  →  09_distributions.png
# Panel 1: Salary histogram  |  Panel 2: Job category tags bar chart
# =============================================================================
def plot_distributions(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel 1: Annual Salary ────────────────────────────────────────────────
    ax = axes[0]
    sal = df["salary_clean"].dropna() if "salary_clean" in df.columns else pd.Series(dtype=float)

    if len(sal) > 0:
        cap  = sal.quantile(0.99)
        data = sal[sal <= cap]
        ax.hist(data, bins=50, color=PALETTE, edgecolor="none", alpha=0.85)
        ax.axvline(sal.median(), color="#e63946", lw=1.8, ls="--",
                   label=f"Median: ${sal.median()/1000:.0f}K")
        ax.axvline(sal.mean(),   color="#f4a261", lw=1.8, ls=":",
                   label=f"Mean:   ${sal.mean()/1000:.0f}K")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
        ax.legend(fontsize=9)
        ax.text(0.97, 0.96,
                f"n = {len(sal):,} postings with salary data\n"
                f"({len(sal)/len(df)*100:.0f}% of all {len(df):,} jobs)",
                ha="right", va="top", fontsize=8.5, color="#666",
                transform=ax.transAxes)

    ax.set_title("Annual Salary Distribution", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Annual Salary (USD)", fontsize=10)
    ax.set_ylabel("Number of job postings", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Panel 2: Job category tags per posting ────────────────────────────────
    ax = axes[1]
    if "skill_count" in df.columns:
        vc = df["skill_count"].value_counts().sort_index()
        bars = ax.bar(
            vc.index.astype(str), vc.values,
            color=PALETTE, edgecolor="none", alpha=0.85, width=0.5
        )
        for bar, val in zip(bars, vc.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + vc.max() * 0.015,
                f"{val:,}\n({val / len(df) * 100:.0f}%)",
                ha="center", va="bottom", fontsize=9.5, color="#333"
            )
        ax.text(0.97, 0.96,
                "Each job is tagged with 0–3\ncategory codes in the skills file",
                ha="right", va="top", fontsize=8.5, color="#666",
                transform=ax.transAxes)

    ax.set_title("Job Category Tags per Posting", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Number of category tags assigned", fontsize=10)
    ax.set_ylabel("Number of job postings", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Exploratory Data Analysis — Key Variable Distributions",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "09_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("STEP 4: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    df = load()
    print_overview(df)
    plot_distributions(df)

    print("\nStep 4 complete. Proceed to: python 05_visualization.py")
