# =============================================================================
# STEP 5 — DATA MINING: CLASSIFICATION
# Rubric: 8.1 Classification
# Task: Predict whether a job is high-demand (applies >= 75th percentile)
# Models: Logistic Regression, Random Forest Classifier, Decision Tree
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize
import joblib
import warnings
warnings.filterwarnings("ignore")

PROCESSED   = "data/processed"
FIGURES_DIR = "outputs/figures"
MODELS_DIR  = "outputs"
os.makedirs(MODELS_DIR, exist_ok=True)


def load():
    path = os.path.join(PROCESSED, "jobs_preprocessed.csv")
    if not os.path.exists(path):
        path = os.path.join(PROCESSED, "jobs_merged.csv")
    return pd.read_csv(path, low_memory=False)


def prepare_features(df):
    """Select and prepare feature matrix X and target y."""

    target = "high_demand"
    if target not in df.columns:
        print("  high_demand column not found — run 03_preprocessing.py first")
        return None, None, None

    # apply_rate = applies / views — including both leaks the target (apply_rate * views = applies).
    # Keep views (job visibility) but drop apply_rate to avoid perfect reconstruction.
    feature_cols = [c for c in [
        "skill_count", "views",
        "experience_encoded", "company_size_encoded",
        "is_remote", "is_senior_title", "high_salary",
    ] if c in df.columns]

    work_cols = [c for c in df.columns if c.startswith("work_")]
    feature_cols += work_cols

    # Restrict to rows where applies data is known (high_demand not NaN)
    df_model = df[df[target].notna()][feature_cols + [target]].copy()
    X = df_model[feature_cols].fillna(0)

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    y = df_model[target].astype(int)
    counts = y.value_counts().to_dict()

    print(f"  Features: {feature_cols}")
    print(f"  Dataset:  {len(X):,} samples (rows with known applies only)")
    print(f"  Class balance: Low-demand={counts.get(0,0):,}  High-demand={counts.get(1,0):,}")
    return X, y, feature_cols


def train_and_evaluate(X, y, feature_cols):
    """Train three classifiers, compare performance."""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
    }

    results = {}
    print("\n  Model Performance Summary")
    print(f"  {'Model':<25} {'Accuracy':>10} {'F1-Score':>10} {'ROC-AUC':>10}")
    print("  " + "-" * 58)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="weighted")
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0

        results[name] = {
            "model": model, "y_pred": y_pred, "y_proba": y_proba,
            "accuracy": acc, "f1": f1, "auc": auc
        }
        print(f"  {name:<25} {acc:>10.3f} {f1:>10.3f} {auc:>10.3f}")

    # Best model detail
    best_name = max(results, key=lambda k: results[k]["f1"])
    best = results[best_name]
    print(f"\n  Best model: {best_name}")
    print("\n" + classification_report(y_test, best["y_pred"],
                                       target_names=["Low-demand", "High-demand"]))

    # Save best model
    joblib.dump(best["model"], os.path.join(MODELS_DIR, "classifier_best.pkl"))

    return results, best_name, X_test, y_test, feature_cols, models["Random Forest"]


def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low-demand", "High-demand"],
                yticklabels=["Low-demand", "High-demand"], ax=ax)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(f"Confusion matrix — {model_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "08_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


_FEATURE_LABELS = {
    "skill_count":          "Skill count",
    "views":                "Job views",
    "apply_rate":           "Apply rate (applies/views)",
    "experience_encoded":   "Experience level",
    "company_size_encoded": "Company size",
    "is_remote":            "Remote job",
    "is_senior_title":      "Senior-level title",
    "high_salary":          "High salary (top 25%)",
    "work_FULL_TIME":       "Work type: Full-time",
    "work_PART_TIME":       "Work type: Part-time",
    "work_CONTRACT":        "Work type: Contract",
    "work_INTERNSHIP":      "Work type: Internship",
    "work_TEMPORARY":       "Work type: Temporary",
    "work_OTHER":           "Work type: Other",
    "work_VOLUNTEER":       "Work type: Volunteer",
}


def plot_feature_importance(rf_model, feature_cols):
    imp = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values()
    imp = imp.tail(12)
    pct = imp / imp.sum() * 100
    labels = [_FEATURE_LABELS.get(c, c.replace("_", " ").title()) for c in imp.index]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels, imp.values, color="#0A66C2", edgecolor="none")

    for bar, p in zip(bars, pct.values):
        w = bar.get_width()
        ax.text(w + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{p:.1f}%", va="center", fontsize=9, color="#333")

    ax.set_title("Which features best predict a high-demand job posting?\n"
                 "(Random Forest — higher bar = more predictive power)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Importance score", fontsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "09_feature_importance_clf.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_roc_curve(results, y_test):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors     = ["#0A66C2", "#e63946", "#2a9d8f"]
    best_fpr   = best_tpr = None
    best_auc   = -1

    for (name, res), color in zip(results.items(), colors):
        if res["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
            ax.plot(fpr, tpr, label=f"{name}  (AUC = {res['auc']:.2f})",
                    color=color, linewidth=2)
            if res["auc"] > best_auc:
                best_auc, best_fpr, best_tpr = res["auc"], fpr, tpr

    # Shade area under best curve
    if best_fpr is not None:
        ax.fill_between(best_fpr, best_tpr, alpha=0.08, color="#0A66C2")

    # Diagonal = random guess baseline
    ax.fill_between([0, 1], [0, 1], alpha=0.04, color="#aaa")
    ax.plot([0, 1], [0, 1], color="#aaa", linewidth=1.5, linestyle="--",
            label="Random guess  (AUC = 0.50)")

    # Annotation: corner = perfect model
    ax.annotate("← Perfect model\n   lives here",
                xy=(0, 1), xytext=(0.12, 0.82),
                fontsize=8, color="#555",
                arrowprops=dict(arrowstyle="->", color="#aaa", lw=1))

    ax.set_xlabel(
        "False Positive Rate\n(fraction of low-demand jobs wrongly predicted as high-demand)",
        fontsize=10)
    ax.set_ylabel(
        "True Positive Rate\n(fraction of high-demand jobs correctly identified)",
        fontsize=10)
    ax.set_title(
        "ROC Curve — predicting high-demand job postings\n"
        "AUC = area under the curve  (closer to 1.0 = better; 0.5 = no better than guessing)",
        fontsize=11, fontweight="bold", pad=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "10_roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 5: CLASSIFICATION — Predict High-Demand Job Postings")
    print("=" * 60)

    df = load()
    X, y, feature_cols = prepare_features(df)

    if X is not None:
        results, best_name, X_test, y_test, feature_cols, rf_model = train_and_evaluate(X, y, feature_cols)
        plot_confusion_matrix(y_test, results[best_name]["y_pred"], best_name)
        plot_feature_importance(rf_model, feature_cols)
        plot_roc_curve(results, y_test)

    print("\nStep 5 complete. Proceed to: python 06_mining_clustering.py")
