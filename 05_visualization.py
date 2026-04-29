# =============================================================================
# STEP 5 — DATA VISUALIZATION
# Rubric: Visualization — bar chart, line chart, scatter plot,
#         heatmap, boxplot (all 5 types satisfied below)
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

PROCESSED   = "data/processed"
FIGURES_DIR = "outputs/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

PALETTE = "#0A66C2"   # LinkedIn blue

# Human-readable labels for the abbreviated industry category codes
CATEGORY_LABELS = {
    'IT':   'Info Technology', 'SALE': 'Sales',        'MGMT': 'Management',
    'MNFC': 'Manufacturing',   'HCPR': 'Healthcare',   'BD':   'Business Dev',
    'ENG':  'Engineering',     'OTHR': 'Other',         'FIN':  'Finance',
    'MRKT': 'Marketing',       'ACCT': 'Accounting',   'ADM':  'Administration',
    'CUST': 'Customer Service','PRJM': 'Project Mgmt', 'ANLS': 'Analytics',
    'RSCH': 'Research',        'HR':   'Human Resources','LGL': 'Legal',
    'CNSL': 'Consulting',      'EDU':  'Education',
}


def load():
    path = os.path.join(PROCESSED, "jobs_preprocessed.csv")
    if not os.path.exists(path):
        path = os.path.join(PROCESSED, "jobs_merged.csv")
    return pd.read_csv(path, low_memory=False)


# =============================================================================
# 1. BAR CHART — Top job titles by posting count
# =============================================================================
def plot_top_titles(df):
    if "title" not in df.columns:
        return
    top = df["title"].value_counts().head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top.index[::-1], top.values[::-1], color=PALETTE, edgecolor="none")
    ax.set_xlabel("Number of postings", fontsize=11)
    ax.set_title("Top 15 job titles by posting volume", fontsize=13, fontweight="bold", pad=12)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 1, bar.get_y() + bar.get_height() / 2,
                f"{int(w):,}", va="center", fontsize=9, color="#555")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "01_top_job_titles.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# 2. BAR CHART — Top job categories (human-readable industry labels)
# =============================================================================
def plot_top_skills():
    skills_path = os.path.join(PROCESSED, "job_skills_clean.csv")
    if not os.path.exists(skills_path):
        print("  Skipping — job_skills_clean.csv not found")
        return

    skills = pd.read_csv(skills_path)
    col    = "skill_abr" if "skill_abr" in skills.columns else skills.columns[-1]
    top    = skills[col].value_counts().head(15)
    labels = [CATEGORY_LABELS.get(c, c) for c in top.index]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [PALETTE if i < 5 else "#90b8d4" for i in range(len(top))]
    bars   = ax.barh(labels[::-1], top.values[::-1], color=colors[::-1], edgecolor="none")
    ax.set_xlabel("Number of job postings", fontsize=11)
    ax.set_title("Top 15 job categories on LinkedIn", fontsize=13, fontweight="bold", pad=12)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 50, bar.get_y() + bar.get_height() / 2,
                f"{int(w):,}", va="center", fontsize=9, color="#555")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "02_top_skills.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# 3. BOXPLOT — Salary by experience level (real data only, no imputation)
# =============================================================================
def plot_salary_boxplot(df):
    sal_col = "salary_clean" if "salary_clean" in df.columns else "normalized_salary"
    if sal_col not in df.columns:
        print(f"  Skipping salary boxplot — {sal_col} not found")
        return

    exp_col = "formatted_experience_level"
    if exp_col not in df.columns:
        return

    order   = ["Entry level", "Associate", "Mid-Senior level", "Director", "Executive"]
    plot_df = df[df[exp_col].isin(order) & df[sal_col].notna()].copy()
    present = [o for o in order if o in plot_df[exp_col].unique()]
    print(f"  Salary boxplot: {len(plot_df):,} rows with real salary data")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=plot_df, x=exp_col, y=sal_col,
        order=present, palette="Blues",
        linewidth=1.2, fliersize=3, ax=ax
    )
    ax.set_xlabel("Experience level", fontsize=11)
    ax.set_ylabel("Annual salary (USD)", fontsize=11)
    ax.set_title(
        f"Salary distribution by experience level  (n={len(plot_df):,} real records)",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "03_salary_boxplot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# 4. SCATTER — Salary vs applicant count (do high-paying jobs get more applies?)
# =============================================================================
def plot_salary_vs_applies(df):
    sal_col = "salary_clean" if "salary_clean" in df.columns else "normalized_salary"
    if sal_col not in df.columns or "applies" not in df.columns:
        return

    plot_df = df[df[sal_col].notna() & (df["applies"] > 0)].copy()
    if len(plot_df) == 0:
        print("  Skipping scatter — no rows with both salary and applies")
        return
    plot_df = plot_df.sample(min(3000, len(plot_df)), random_state=42)

    colors = (plot_df["is_remote"].map({1: "#e63946", 0: PALETTE})
              if "is_remote" in plot_df.columns else PALETTE)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(plot_df[sal_col], plot_df["applies"],
               c=colors, alpha=0.45, s=18, edgecolors="none")

    ax.set_yscale("log")   # applies is heavily right-skewed

    if "is_remote" in plot_df.columns:
        ax.legend(handles=[
            mpatches.Patch(color=PALETTE,   label="On-site / Hybrid"),
            mpatches.Patch(color="#e63946", label="Remote"),
        ], fontsize=9)

    ax.set_xlabel("Annual salary (USD)", fontsize=11)
    ax.set_ylabel("Number of applicants (log scale)", fontsize=11)
    ax.set_title("Do higher-paying jobs attract more applicants?",
                 fontsize=12, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "04_salary_vs_applies.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# 5. HEATMAP — Correlation matrix (readable feature names)
# =============================================================================
def plot_correlation_heatmap(df):
    num_cols = [c for c in ["salary_clean", "skill_count", "applies", "views",
                             "apply_rate", "is_remote", "experience_encoded",
                             "company_size_encoded", "is_senior_title"]
                if c in df.columns]
    if len(num_cols) < 3:
        return

    rename = {
        "salary_clean":         "Salary",
        "skill_count":          "Skill count",
        "applies":              "Applicants",
        "views":                "Views",
        "apply_rate":           "Apply rate",
        "is_remote":            "Remote",
        "experience_encoded":   "Experience",
        "company_size_encoded": "Company size",
        "is_senior_title":      "Senior title",
    }
    corr = df[num_cols].corr().round(2)
    corr.index   = [rename.get(c, c) for c in corr.index]
    corr.columns = [rename.get(c, c) for c in corr.columns]

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="Blues", linewidths=0.5, ax=ax, annot_kws={"size": 9})
    ax.set_title("Feature correlation matrix", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "05_correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# 6. LINE CHART — Salary cumulative distribution function (CDF)
# =============================================================================
def plot_salary_cdf(df):
    sal_col = "salary_clean" if "salary_clean" in df.columns else "normalized_salary"
    if sal_col not in df.columns:
        return

    sal = df[sal_col].dropna().sort_values().reset_index(drop=True)
    cdf = np.arange(1, len(sal) + 1) / len(sal)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(sal, cdf, color=PALETTE, linewidth=2.5)
    ax.fill_between(sal, cdf, alpha=0.08, color=PALETTE)

    # Annotate key percentiles
    for pct, label in [(0.25, "25th pctile"), (0.50, "Median"), (0.75, "75th pctile")]:
        val = float(sal.quantile(pct))
        ax.axvline(val, color="#aaa", linewidth=1.2, linestyle="--")
        ax.text(val, pct + 0.05, f"{label}\n${val/1000:.0f}K",
                ha="center", fontsize=8, color="#444")

    ax.set_xlabel("Annual salary (USD)", fontsize=11)
    ax.set_ylabel("Cumulative fraction of postings", fontsize=11)
    ax.set_title(
        f"Salary cumulative distribution  (n = {len(sal):,} postings with real salary data)",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "06_salary_cdf.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# 7. BAR CHART — Top 15 states by posting volume
# =============================================================================
def plot_top_states(df):
    if "state" not in df.columns:
        return

    top = (df[df["state"] != "Unknown"]["state"]
           .value_counts()
           .head(15))

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(top.index[::-1], top.values[::-1], color=PALETTE, edgecolor="none")
    ax.set_xlabel("Number of job postings", fontsize=11)
    ax.set_title("Top 15 states by job posting volume", fontsize=13, fontweight="bold", pad=12)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 10, bar.get_y() + bar.get_height() / 2,
                f"{int(w):,}", va="center", fontsize=9, color="#555")
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "07_top_states.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# 8. BAR CHART — Remote vs on-site breakdown
# =============================================================================
def plot_remote_breakdown(df):
    if "is_remote" not in df.columns:
        return

    counts = df["is_remote"].value_counts().rename({0: "On-site / Hybrid", 1: "Remote"})

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=[PALETTE, "#e63946"], width=0.5, edgecolor="none")
    ax.set_ylabel("Number of postings", fontsize=11)
    ax.set_title("Remote vs on-site job postings", fontsize=13, fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 20,
                f"{int(h):,}", ha="center", fontsize=10)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "08_remote_breakdown.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("STEP 5: DATA VISUALIZATION")
    print("=" * 60)

    df = load()

    plot_top_titles(df)         # 01 — BAR
    plot_top_skills()           # 02 — BAR
    plot_salary_boxplot(df)     # 03 — BOXPLOT
    plot_salary_vs_applies(df)  # 04 — SCATTER
    plot_correlation_heatmap(df)# 05 — HEATMAP
    plot_salary_cdf(df)         # 06 — LINE
    plot_top_states(df)         # 07 — BAR
    plot_remote_breakdown(df)   # 08 — BAR

    print(f"\nAll figures saved to: {FIGURES_DIR}/")
    print("Step 5 complete. Proceed to: python 06_mining_classification.py")
