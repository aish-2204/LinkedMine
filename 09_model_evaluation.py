# =============================================================================
# STEP 9 — MODEL EVALUATION (Consolidated Report)
# Rubric: 9. Model Evaluation
# Covers: Accuracy, F1, ROC-AUC (classification)
#         Silhouette (clustering)
#         Support/Confidence/Lift (association rules)
#         RMSE, MAE, R² (regression)
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
OUTPUTS_DIR = "outputs"


def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================================================
# RE-RUN QUICK EVALUATION — loads saved models or re-trains
# =============================================================================

def evaluate_classification():
    print_header("CLASSIFICATION — High-Demand Job Postings")
    try:
        import joblib
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
        from sklearn.model_selection import train_test_split

        df = pd.read_csv(os.path.join(PROCESSED, "jobs_preprocessed.csv"), low_memory=False)
        model = joblib.load(os.path.join(OUTPUTS_DIR, "classifier_best.pkl"))

        feature_cols = [c for c in ["skill_count","views",
                                     "experience_encoded","company_size_encoded",
                                     "is_remote","is_senior_title","high_salary"] if c in df.columns]
        work_cols = [c for c in df.columns if c.startswith("work_")]
        feature_cols += work_cols

        df_m = df[feature_cols + ["high_demand"]].dropna()
        X = df_m[feature_cols].fillna(0)
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
        y = df_m["high_demand"].astype(int)
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="weighted")
        auc = roc_auc_score(y_test, y_proba)

        print(f"  Accuracy  : {acc:.4f}")
        print(f"  F1-Score  : {f1:.4f}")
        print(f"  ROC-AUC   : {auc:.4f}")
        print("\n  Per-class report:")
        report = classification_report(y_test, y_pred, target_names=["Low-demand","High-demand"])
        print("\n".join("    " + l for l in report.splitlines()))
        return {"accuracy": acc, "f1": f1, "roc_auc": auc}
    except Exception as e:
        print(f"  [Skipped — run 05 first] {e}")
        return {}


def evaluate_clustering():
    print_header("CLUSTERING — Job Role Families")
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        from sklearn.preprocessing import MultiLabelBinarizer

        skills = pd.read_csv(os.path.join(PROCESSED, "job_skills_clean.csv"))
        col = "skill_abr" if "skill_abr" in skills.columns else skills.columns[-1]
        jcol = "job_id"   if "job_id"   in skills.columns else skills.columns[0]

        top_skills = skills[col].value_counts().head(50).index.tolist()
        filtered   = skills[skills[col].isin(top_skills)]
        job_skills = filtered.groupby(jcol)[col].apply(list).reset_index()
        job_skills.columns = ["job_id","skills"]

        mlb = MultiLabelBinarizer(classes=top_skills)
        X   = mlb.fit_transform(job_skills["skills"])

        best_sil, best_k = -1, 3
        for k in range(2, 9):
            km     = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            sil    = silhouette_score(X, labels, sample_size=min(2000, len(X)))
            if sil > best_sil:
                best_sil, best_k = sil, k

        km     = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil    = silhouette_score(X, labels, sample_size=min(2000, len(X)))
        dbi    = davies_bouldin_score(X.toarray() if hasattr(X,"toarray") else X, labels)

        print(f"  Optimal K          : {best_k}")
        print(f"  Silhouette score   : {sil:.4f}  (closer to 1.0 = better)")
        print(f"  Davies-Bouldin idx : {dbi:.4f}  (lower = better)")
        print(f"  Cluster sizes:")
        unique, counts = np.unique(labels, return_counts=True)
        for c, n in zip(unique, counts):
            print(f"    Cluster {c+1}: {n:,} jobs ({n/len(labels)*100:.1f}%)")

        return {"optimal_k": best_k, "silhouette": sil, "davies_bouldin": dbi}
    except Exception as e:
        print(f"  [Skipped — run 06 first] {e}")
        return {}


def evaluate_association_rules():
    print_header("ASSOCIATION RULES — Skill Co-occurrence")
    try:
        rules = pd.read_csv(os.path.join(OUTPUTS_DIR, "association_rules.csv"))

        print(f"  Total rules generated   : {len(rules):,}")
        print(f"  Support  range          : {rules['support'].min():.3f} – {rules['support'].max():.3f}")
        print(f"  Confidence range        : {rules['confidence'].min():.3f} – {rules['confidence'].max():.3f}")
        print(f"  Lift range              : {rules['lift'].min():.2f} – {rules['lift'].max():.2f}")
        print(f"\n  Rules with Lift > 1.5   : {(rules['lift'] > 1.5).sum():,}")
        print(f"  Rules with Conf  > 0.5  : {(rules['confidence'] > 0.5).sum():,}")

        print(f"\n  Top 5 rules by Lift:")
        print(f"  {'Antecedent':<30} {'Consequent':<25} {'Support':>8} {'Conf':>8} {'Lift':>7}")
        print("  " + "-" * 82)
        for _, r in rules.head(5).iterrows():
            print(f"  {str(r['antecedents']):<30} → {str(r['consequents']):<25}"
                  f"  {r['support']:>7.3f}  {r['confidence']:>7.3f}  {r['lift']:>6.2f}")

        return {
            "total_rules": len(rules),
            "max_lift":    rules["lift"].max(),
            "avg_confidence": rules["confidence"].mean()
        }
    except Exception as e:
        print(f"  [Skipped — run 07 first] {e}")
        return {}


def evaluate_regression():
    print_header("REGRESSION — Salary Prediction (salary_clean)")
    try:
        import joblib
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from sklearn.model_selection import train_test_split

        df    = pd.read_csv(os.path.join(PROCESSED, "jobs_preprocessed.csv"), low_memory=False)
        model = joblib.load(os.path.join(OUTPUTS_DIR, "regressor_best.pkl"))

        feature_cols = [c for c in ["skill_count","applies","views","apply_rate",
                                     "experience_encoded","company_size_encoded",
                                     "is_remote","is_senior_title"] if c in df.columns]
        work_cols = [c for c in df.columns if c.startswith("work_")]
        feature_cols += work_cols

        target = "salary_clean"
        df_m = df[feature_cols + [target]].dropna(subset=[target]).copy()
        df_m[feature_cols] = df_m[feature_cols].fillna(0)
        X = df_m[feature_cols].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
        y = df_m[target]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        y_pred = model.predict(X_test)
        rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
        mae    = mean_absolute_error(y_test, y_pred)
        r2     = r2_score(y_test, y_pred)
        mape   = np.mean(np.abs((y_test - y_pred) / y_test.replace(0, np.nan))) * 100

        print(f"  RMSE    : ${rmse:>10,.0f}  (avg prediction error)")
        print(f"  MAE     : ${mae:>10,.0f}  (mean absolute error)")
        print(f"  R²      : {r2:>10.4f}  (variance explained)")
        print(f"  MAPE    : {mape:>10.1f}%  (mean absolute % error)")

        return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}
    except Exception as e:
        print(f"  [Skipped — run 08 first] {e}")
        return {}


def plot_summary_scorecard(clf, clu, arm, reg):
    """Visual scorecard — one panel per mining task with progress bars and interpretation."""

    PANELS = [
        {
            "title": "Classification",
            "subtitle": "Predict high-demand job postings",
            "color": "#0A66C2",
            "rows": [
                ("Accuracy",  clf.get("accuracy", 0),  1.0, "{:.1%}", "fraction of correct predictions"),
                ("F1 Score",  clf.get("f1",       0),  1.0, "{:.3f}", "balance of precision & recall"),
                ("ROC-AUC",   clf.get("roc_auc",   0), 1.0, "{:.3f}", "ability to rank high vs low demand"),
            ] if clf else [],
        },
        {
            "title": "Clustering",
            "subtitle": "Discover job role families (K-Means)",
            "color": "#2a9d8f",
            "rows": [
                ("Optimal K",        clu.get("optimal_k",     0),   10, "{}",     "number of role families found"),
                ("Silhouette",       clu.get("silhouette",    0),    1.0, "{:.3f}", "how distinct the clusters are (1=perfect)"),
                ("Davies-Bouldin",   clu.get("davies_bouldin",9),   None, "{:.3f}", "cluster separation (lower = better)"),
            ] if clu else [],
        },
        {
            "title": "Association Rules",
            "subtitle": "Which job categories co-occur? (Apriori)",
            "color": "#f4a261",
            "rows": [
                ("Total rules",     arm.get("total_rules",  0),    None, "{}",    "skill co-occurrence rules found"),
                ("Max lift",        arm.get("max_lift",     0),    None, "{:.2f}", "strongest co-occurrence signal"),
                ("Avg confidence",  arm.get("avg_confidence",0),   1.0,  "{:.3f}", "avg rule reliability"),
            ] if arm else [],
        },
        {
            "title": "Regression",
            "subtitle": "Predict annual salary (salary_clean)",
            "color": "#e63946",
            "rows": [
                ("R²",    reg.get("r2",   0),   1.0,  "{:.3f}", f"explains {reg.get('r2',0)*100:.0f}% of salary variation"),
                ("RMSE",  reg.get("rmse", 0),   None, "${:,.0f}", "avg dollar prediction error"),
                ("MAE",   reg.get("mae",  0),   None, "${:,.0f}", "median absolute prediction error"),
                ("MAPE",  reg.get("mape", 0),   None, "{:.1f}%",  "mean absolute % error"),
            ] if reg else [],
        },
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Mining the Job Market — Model Evaluation Summary",
                 fontsize=14, fontweight="bold", y=1.01)

    for ax, panel in zip(axes.flat, PANELS):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        color = panel["color"]

        # Panel border
        rect = plt.Rectangle((0.01, 0.01), 0.98, 0.98, fill=False,
                              edgecolor=color, linewidth=2, transform=ax.transAxes)
        ax.add_patch(rect)

        # Header
        ax.text(0.5, 0.95, panel["title"],  ha="center", va="top",
                fontsize=13, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(0.5, 0.88, panel["subtitle"], ha="center", va="top",
                fontsize=9,  color="#777", style="italic", transform=ax.transAxes)
        ax.axhline(0.84, xmin=0.05, xmax=0.95, color=color, linewidth=0.8, alpha=0.4)

        if not panel["rows"]:
            ax.text(0.5, 0.5, "Run the corresponding\nmining script first",
                    ha="center", va="center", fontsize=10, color="#aaa",
                    transform=ax.transAxes)
            continue

        y_pos = 0.76
        row_h = (y_pos - 0.06) / len(panel["rows"])

        for metric_name, value, max_val, fmt, explanation in panel["rows"]:
            try:
                val_str = fmt.format(value)
            except Exception:
                val_str = str(value)

            # Metric label
            ax.text(0.07, y_pos, metric_name, fontsize=10, color="#333",
                    va="center", transform=ax.transAxes)
            # Value
            ax.text(0.93, y_pos, val_str, fontsize=11, fontweight="bold",
                    color=color, ha="right", va="center", transform=ax.transAxes)

            # Progress bar (only when max_val is defined and value is a fraction/score)
            if max_val and isinstance(value, (int, float)) and max_val > 0:
                bar_w  = min(1.0, float(value) / float(max_val)) * 0.55
                bar_bg = plt.Rectangle((0.07, y_pos - 0.027), 0.55, 0.022,
                                       color="#eee", transform=ax.transAxes)
                bar_fg = plt.Rectangle((0.07, y_pos - 0.027), bar_w, 0.022,
                                       color=color, alpha=0.5, transform=ax.transAxes)
                ax.add_patch(bar_bg)
                ax.add_patch(bar_fg)

            # Explanation sub-label
            ax.text(0.07, y_pos - 0.055, explanation, fontsize=7.5, color="#999",
                    va="center", style="italic", transform=ax.transAxes)

            ax.axhline(y_pos - row_h + 0.01, xmin=0.04, xmax=0.96,
                       color="#eee", linewidth=0.5)
            y_pos -= row_h

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "20_evaluation_scorecard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Scorecard saved → {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 9: CONSOLIDATED MODEL EVALUATION REPORT")
    print("=" * 60)

    clf = evaluate_classification()
    clu = evaluate_clustering()
    arm = evaluate_association_rules()
    reg = evaluate_regression()

    plot_summary_scorecard(clf, clu, arm, reg)

    print("\n" + "=" * 60)
    print("  ALL STEPS COMPLETE")
    print("=" * 60)
    print("  Figures     → outputs/figures/")
    print("  Models      → outputs/*.pkl")
    print("  Rules       → outputs/association_rules.csv")
    print("  Warehouse   → data/warehouse/etl_warehouse.db")
    print("  Clean data  → data/processed/jobs_preprocessed.csv")
    print("\n  Project title:")
    print('  "Mining the job market — skills, salaries, and hiring')
    print('   patterns from 33,000 LinkedIn postings"')
