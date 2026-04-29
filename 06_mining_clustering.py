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
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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

# Distinct colors, one per cluster (supports up to 5)
CLUSTER_COLORS = ["#0A66C2", "#2a9d8f", "#e63946", "#f4a261", "#8338ec"]


def load():
    skills_path = os.path.join(PROCESSED, "job_skills_clean.csv")
    jobs_path   = os.path.join(PROCESSED, "jobs_preprocessed.csv")
    if not os.path.exists(jobs_path):
        jobs_path = os.path.join(PROCESSED, "jobs_merged.csv")

    jobs   = pd.read_csv(jobs_path,   low_memory=False)
    skills = pd.read_csv(skills_path, low_memory=False) if os.path.exists(skills_path) else None
    return jobs, skills


def build_skill_matrix(jobs, skills):
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


def find_optimal_k(X, k_range=range(2, 6)):
    """Find best K in 2–5 — capped so clusters stay distinct and interpretable."""
    print("  Evaluating K (2–5) via silhouette score...")
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


def _auto_name(center, skill_names, threshold=0.12, max_terms=2):
    """Generate a cluster name from its top dominant categories."""
    top_idx = np.argsort(center)[::-1]
    sig = [i for i in top_idx if center[i] >= threshold][:max_terms]
    if not sig:
        sig = top_idx[:1]
    return " & ".join(CATEGORY_LABELS.get(skill_names[i], skill_names[i]) for i in sig)


def plot_cluster_scatter(X, labels, top_skills, k, n_skill_cols):
    """PCA scatter: each dot = one job posting, colour = cluster, ★ = centroid."""
    X_skill     = X[:, :n_skill_cols]
    skill_names = top_skills[:n_skill_cols]

    # Auto-name every cluster
    names = []
    sizes = []
    for c in range(k):
        mask = labels == c
        sizes.append(int(mask.sum()))
        center = X_skill[mask].mean(axis=0) if mask.sum() > 0 else np.zeros(len(skill_names))
        names.append(_auto_name(center, skill_names))

    # PCA on the full feature matrix (skill binary + numeric supplement)
    pca  = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    var  = pca.explained_variance_ratio_

    # Project K-Means centroids into the same PCA space
    # Refit KMeans to get .cluster_centers_, then transform
    km_refit = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_refit.fit(X)
    centroids_2d = pca.transform(km_refit.cluster_centers_)

    # Sample points to keep the plot readable (avoid solid blobs)
    n_sample = min(10_000, len(X_2d))
    rng      = np.random.default_rng(42)
    idx      = rng.choice(len(X_2d), n_sample, replace=False)

    colors = CLUSTER_COLORS[:k]

    fig, ax = plt.subplots(figsize=(11, 7))

    # ── Draw each cluster's points ────────────────────────────────────────────
    for c in range(k):
        mask_s = labels[idx] == c
        ax.scatter(
            X_2d[idx][mask_s, 0],
            X_2d[idx][mask_s, 1],
            c=colors[c], alpha=0.30, s=14,
            edgecolors="none",
            label=f"{names[c]}  (n={sizes[c]:,})"
        )

    # ── Draw centroids ────────────────────────────────────────────────────────
    for c in range(k):
        cx, cy = centroids_2d[c]
        ax.scatter(cx, cy,
                   c=colors[c], s=350, marker="*",
                   edgecolors="white", linewidths=1.5, zorder=6)
        ax.annotate(
            names[c],
            xy=(cx, cy),
            xytext=(12, 10), textcoords="offset points",
            fontsize=9, fontweight="bold", color=colors[c],
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", edgecolor=colors[c],
                      alpha=0.88, linewidth=1.2),
            zorder=7
        )

    ax.set_xlabel(f"PC1  ({var[0]*100:.1f}% of variance)", fontsize=11)
    ax.set_ylabel(f"PC2  ({var[1]*100:.1f}% of variance)", fontsize=11)
    ax.set_title(
        "Job Role Clusters  —  each dot is one job posting\n"
        "★ = cluster centroid  |  colour = role family",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.legend(fontsize=8.5, loc="upper right",
              framealpha=0.92, edgecolor="#ddd",
              title="Cluster (role family)", title_fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

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
    print("  " + "-" * 70)
    for c in range(k):
        mask    = labels == c
        center  = X_skill[mask].mean(axis=0)
        top_idx = np.argsort(center)[::-1][:top_n]
        cats    = [f"{CATEGORY_LABELS.get(skill_names[i], skill_names[i])} ({center[i]:.0%})"
                   for i in top_idx]
        name    = _auto_name(center, skill_names)
        pct     = mask.sum() / len(labels) * 100
        print(f"  [{name}]  {pct:.1f}%  n={mask.sum():,}")
        print(f"    → {', '.join(cats)}")


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

    best_k          = find_optimal_k(X)
    labels, km, sil = run_kmeans(X, best_k)

    plot_cluster_scatter(X, labels, top_skills, best_k, n_skill_cols)
    describe_clusters(X, labels, top_skills, best_k, n_skill_cols)

    print(f"\n  Final silhouette score: {sil:.4f}")
    print("\nStep 6 complete. Proceed to: python 07_mining_association_rules.py")
