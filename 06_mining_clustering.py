# =============================================================================
# STEP 6 — DATA MINING: CLUSTERING
# Rubric: 8.2 Clustering
# Task: Group job postings into natural role families based on skill profiles
# Method: K-Means on job-category binary matrix + numeric job attributes
# Evaluation: Silhouette score (computed internally, no plot)
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

PROCESSED   = "data/processed"
FIGURES_DIR = "outputs/figures"

CATEGORY_LABELS = {
    'IT':   'Info Technology', 'SALE': 'Sales',        'MGMT': 'Management',
    'MNFC': 'Manufacturing',   'HCPR': 'Healthcare',   'BD':   'Business Dev',
    'ENG':  'Engineering',     'OTHR': 'Other',         'FIN':  'Finance',
    'MRKT': 'Marketing',       'ACCT': 'Accounting',   'ADM':  'Administration',
    'CUST': 'Customer Service','PRJM': 'Project Mgmt', 'ANLS': 'Analytics',
    'RSCH': 'Research',        'HR':   'Human Res',     'LGL':  'Legal',
    'CNSL': 'Consulting',      'EDU':  'Education',     'SCI':  'Science',
    'DSGN': 'Design',          'WRT':  'Writing',       'TRNG': 'Training',
    'PROD': 'Product',         'SUPL': 'Supply Chain',  'ADVR': 'Advertising',
    'STRA': 'Strategy',        'GENB': 'General Business',
}


def load():
    skills_path = os.path.join(PROCESSED, "job_skills_clean.csv")
    jobs_path   = os.path.join(PROCESSED, "jobs_preprocessed.csv")
    if not os.path.exists(jobs_path):
        jobs_path = os.path.join(PROCESSED, "jobs_merged.csv")

    jobs   = pd.read_csv(jobs_path,   low_memory=False)
    skills = pd.read_csv(skills_path, low_memory=False) if os.path.exists(skills_path) else None
    return jobs, skills


def build_skill_matrix(jobs, skills):
    """Build feature matrix: job-category binary + scaled numeric attributes.

    Returns X, job_ids, skill_names, n_skill_cols
    where the first n_skill_cols columns are the binary skill features.
    """
    if skills is None:
        print("  No skills file — using numeric features for clustering")
        return None, None, None, 0

    skill_col = "skill_abr" if "skill_abr" in skills.columns else skills.columns[-1]
    job_col   = "job_id"    if "job_id"    in skills.columns else skills.columns[0]

    job_skills = skills.groupby(job_col)[skill_col].apply(list).reset_index()
    job_skills.columns = ["job_id", "skills"]

    top_skills = skills[skill_col].value_counts().head(35).index.tolist()
    job_skills["skills_filtered"] = job_skills["skills"].apply(
        lambda s: [x for x in s if x in top_skills]
    )
    job_skills = job_skills[job_skills["skills_filtered"].str.len() > 0]

    mlb      = MultiLabelBinarizer(classes=top_skills)
    X_skill  = mlb.fit_transform(job_skills["skills_filtered"])
    job_ids  = job_skills["job_id"].values
    n_skills = X_skill.shape[1]

    # Supplement with scaled numeric features for better cluster separation
    num_cols = [c for c in ["experience_encoded", "company_size_encoded", "is_remote"]
                if c in jobs.columns]
    if num_cols and "job_id" in jobs.columns:
        jobs_idx = jobs.set_index("job_id")
        valid    = [jid for jid in job_ids if jid in jobs_idx.index]
        if len(valid) == len(job_ids):
            num_data   = jobs_idx.loc[job_ids, num_cols].fillna(0).values.astype(float)
            num_scaled = MinMaxScaler().fit_transform(num_data) * 0.5
            X_combined = np.hstack([X_skill, num_scaled])
            print(f"  Feature matrix: {X_combined.shape}  "
                  f"({n_skills} category cols + {len(num_cols)} numeric cols)")
            return X_combined, job_ids, top_skills, n_skills

    print(f"  Skill matrix: {X_skill.shape}")
    return X_skill, job_ids, top_skills, n_skills


def find_optimal_k(X, k_range=range(2, 11)):
    """Find best K by silhouette score — no plot generated."""
    print("  Evaluating K (2–10) via silhouette score...")
    sample_X = X if len(X) <= 5000 else X[np.random.choice(len(X), 5000, replace=False)]

    best_k, best_sil = 3, -1
    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(sample_X)
        sil    = silhouette_score(sample_X, labels, sample_size=min(2000, len(sample_X)))
        print(f"    K={k}  silhouette={sil:.3f}")
        if sil > best_sil:
            best_sil, best_k = sil, k

    print(f"  → Best K={best_k}  silhouette={best_sil:.3f}")
    return best_k


def run_kmeans(X, k):
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil    = silhouette_score(X, labels, sample_size=min(2000, len(X)))
    print(f"\n  K-Means (K={k}): silhouette = {sil:.4f}")
    return labels, km, sil


def plot_cluster_profiles(X, labels, top_skills, k, n_skill_cols):
    """Heatmap: clusters × top job categories.

    Each cell shows the fraction of jobs in that cluster that belong to
    a given category — immediately shows what each cluster is about.
    """
    # Use only the skill binary columns (not the appended numeric features)
    X_skill = X[:, :n_skill_cols]
    skill_names = top_skills[:n_skill_cols]

    # Pick top 15 categories by overall frequency
    overall_freq = X_skill.mean(axis=0)
    top_idx      = np.argsort(overall_freq)[::-1][:15]
    col_names    = [CATEGORY_LABELS.get(skill_names[i], skill_names[i]) for i in top_idx]

    # Build cluster × category frequency matrix
    sizes   = []
    profile = np.zeros((k, len(top_idx)))
    for c in range(k):
        mask = labels == c
        sizes.append(mask.sum())
        if mask.sum() > 0:
            profile[c] = X_skill[mask][:, top_idx].mean(axis=0)

    row_labels = [f"Cluster {c+1}  (n={sizes[c]:,})" for c in range(k)]

    fig, ax = plt.subplots(figsize=(14, max(5, k * 0.65 + 1)))
    sns.heatmap(
        profile,
        xticklabels=col_names,
        yticklabels=row_labels,
        annot=True, fmt=".2f",
        cmap="Blues",
        linewidths=0.4,
        ax=ax,
        annot_kws={"size": 8},
        vmin=0, vmax=profile.max()
    )
    ax.set_title(
        "Job role cluster profiles\n"
        "Each cell = fraction of jobs in that cluster belonging to the category  "
        "(darker = more dominant)",
        fontsize=11, fontweight="bold", pad=14
    )
    ax.set_xlabel("Job category", fontsize=10)
    ax.set_ylabel("Cluster", fontsize=10)
    ax.tick_params(axis="x", rotation=40)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "12_cluster_profiles.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def describe_clusters(X, labels, top_skills, k, n_skill_cols, top_n=5):
    """Print the top categories defining each cluster."""
    X_skill     = X[:, :n_skill_cols]
    skill_names = top_skills[:n_skill_cols]

    print(f"\n  Cluster role families (top {top_n} categories):")
    print("  " + "-" * 60)
    for c in range(k):
        mask    = labels == c
        center  = X_skill[mask].mean(axis=0)
        top_idx = np.argsort(center)[::-1][:top_n]
        cats    = [CATEGORY_LABELS.get(skill_names[i], skill_names[i]) for i in top_idx]
        pct     = mask.sum() / len(labels) * 100
        print(f"  Cluster {c+1:>2} ({pct:4.1f}%  n={mask.sum():,}):  {', '.join(cats)}")


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 6: CLUSTERING — Discover job role families")
    print("=" * 60)

    jobs, skills = load()
    X, job_ids, top_skills, n_skill_cols = build_skill_matrix(jobs, skills)

    if X is None:
        num_cols = [c for c in ["skill_count", "applies", "views", "experience_encoded",
                                 "company_size_encoded", "salary_clean"] if c in jobs.columns]
        X = jobs[num_cols].fillna(0).values
        top_skills   = num_cols
        n_skill_cols = len(num_cols)
        print(f"  Numeric feature matrix: {X.shape}")

    best_k         = find_optimal_k(X)
    labels, km, sil = run_kmeans(X, best_k)

    plot_cluster_profiles(X, labels, top_skills, best_k, n_skill_cols)
    describe_clusters(X, labels, top_skills, best_k, n_skill_cols)

    print(f"\n  Final silhouette score: {sil:.4f}")
    print("\nStep 6 complete. Proceed to: python 07_mining_association_rules.py")
