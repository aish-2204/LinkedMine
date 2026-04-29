# =============================================================================
# STEP 9 — DATA MINING: REGRESSION
# Rubric: 8.4 Regression
# Task: Predict salary_clean (real salary from normalized_salary, $20K–$500K)
# Models: Linear Regression, Ridge, Random Forest, Gradient Boosting
# Evaluation: RMSE, MAE, R²
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings("ignore")

PROCESSED   = "data/processed"
FIGURES_DIR = "outputs/figures"
MODELS_DIR  = "outputs"

_FEATURE_LABELS = {
    "skill_count":          "Skill count",
    "applies":              "Applicant count",
    "views":                "Job views",
    "apply_rate":           "Apply rate",
    "experience_encoded":   "Experience level",
    "company_size_encoded": "Company size",
    "is_remote":            "Remote job",
    "is_senior_title":      "Senior-level title",
    "work_FULL_TIME":       "Full-time",
    "work_PART_TIME":       "Part-time",
    "work_CONTRACT":        "Contract",
    "work_INTERNSHIP":      "Internship",
    "work_TEMPORARY":       "Temporary",
    "work_OTHER":           "Other work type",
    "work_VOLUNTEER":       "Volunteer",
}


def load():
    path = os.path.join(PROCESSED, "jobs_preprocessed.csv")
    if not os.path.exists(path):
        path = os.path.join(PROCESSED, "jobs_merged.csv")
    return pd.read_csv(path, low_memory=False)


def prepare_features(df):
    """Predict salary_clean — real salary data only, no imputed rows."""
    target = "salary_clean"
    if target not in df.columns:
        print("  salary_clean not found — run 03_preprocessing.py first")
        return None, None, None

    feature_cols = [c for c in [
        "skill_count", "applies", "views", "apply_rate",
        "experience_encoded", "company_size_encoded",
        "is_remote", "is_senior_title",
    ] if c in df.columns]

    work_cols = [c for c in df.columns if c.startswith("work_")]
    feature_cols += work_cols

    # Only rows with real salary data
    df_model = df[feature_cols + [target]].dropna(subset=[target]).copy()
    df_model[feature_cols] = df_model[feature_cols].fillna(0)

    X = df_model[feature_cols]
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    y = df_model[target]

    print(f"  Rows with real salary: {len(X):,}")
    print(f"  Salary range: ${y.min():,.0f} – ${y.max():,.0f}  (median: ${y.median():,.0f})")
    print(f"  Features: {feature_cols}")
    return X, y, feature_cols


def train_and_evaluate(X, y, feature_cols):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression":       LinearRegression(),
        "Ridge Regression":        Ridge(alpha=1.0),
        "Random Forest":           RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "Gradient Boosting":       GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
    }

    results = {}
    print("\n  Model Performance Summary")
    print(f"  {'Model':<25} {'RMSE':>12} {'MAE':>12} {'R²':>8}")
    print("  " + "-" * 60)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)

        results[name] = {"model": model, "y_pred": y_pred, "rmse": rmse, "mae": mae, "r2": r2}
        print(f"  {name:<25} ${rmse:>11,.0f} ${mae:>11,.0f} {r2:>8.3f}")

    best_name = max(results, key=lambda k: results[k]["r2"])
    print(f"\n  Best model: {best_name}  (R² = {results[best_name]['r2']:.3f})")
    joblib.dump(results[best_name]["model"], os.path.join(MODELS_DIR, "regressor_best.pkl"))

    return results, best_name, X_test, y_test


def plot_actual_vs_predicted(y_test, y_pred, model_name, r2):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, y_pred, alpha=0.4, s=15, color="#0A66C2", edgecolors="none")

    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")

    # Annotate R² on the plot
    ax.text(0.05, 0.92, f"R² = {r2:.3f}", transform=ax.transAxes,
            fontsize=11, color="#0A66C2", fontweight="bold")

    ax.set_xlabel("Actual salary (USD)", fontsize=11)
    ax.set_ylabel("Predicted salary (USD)", fontsize=11)
    ax.set_title(f"Actual vs Predicted salary\nModel: {model_name}",
                 fontsize=12, fontweight="bold", pad=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "16_actual_vs_predicted.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_residuals(y_test, y_pred, model_name):
    residuals = y_test.values - y_pred
    mean_res  = residuals.mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.scatter(y_pred, residuals, alpha=0.4, s=12, color="#0A66C2", edgecolors="none")
    ax1.axhline(0,        color="#e63946", linewidth=1.5, linestyle="--", label="Zero error")
    ax1.axhline(mean_res, color="#f4a261", linewidth=1.2, linestyle=":",
                label=f"Mean residual ${mean_res:+,.0f}")
    ax1.set_xlabel("Predicted salary", fontsize=11)
    ax1.set_ylabel("Residual  (actual − predicted)", fontsize=11)
    ax1.set_title("Residual plot\n(random scatter around zero = good fit)",
                  fontsize=11, fontweight="bold")
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax1.legend(fontsize=8)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.hist(residuals, bins=40, color="#0A66C2", edgecolor="white", linewidth=0.5)
    ax2.axvline(0,        color="#e63946", linewidth=1.5, linestyle="--", label="Zero residual")
    ax2.axvline(mean_res, color="#f4a261", linewidth=1.2, linestyle=":",
                label=f"Mean ${mean_res:+,.0f}")
    ax2.set_xlabel("Residual value (USD)", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.set_title("Residual distribution\n(bell curve centred on 0 = unbiased model)",
                  fontsize=11, fontweight="bold")
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax2.legend(fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.suptitle(f"Residual analysis — {model_name}", fontsize=11, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "17_residual_plots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_importance_reg(rf_model, feature_cols):
    imp = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values()
    pct = imp / imp.sum() * 100
    labels = [_FEATURE_LABELS.get(c, c.replace("_", " ").title()) for c in imp.index]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels, imp.values, color="#0A66C2", edgecolor="none")

    for bar, p in zip(bars, pct.values):
        w = bar.get_width()
        ax.text(w + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{p:.1f}%", va="center", fontsize=9, color="#333")

    ax.set_title("Which features best predict salary?\n"
                 "(Random Forest — higher bar = stronger predictor)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Importance score", fontsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "18_feature_importance_reg.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_model_comparison(results, y_test):
    # Include a naïve baseline (always predict mean salary) for context
    naive_rmse = float(np.sqrt(np.mean((y_test.values - y_test.mean()) ** 2)))
    all_names  = ["Naïve\n(always predict mean)"] + list(results.keys())
    all_r2s    = [0.0]        + [results[n]["r2"]   for n in results]
    all_rmses  = [naive_rmse] + [results[n]["rmse"] for n in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── R² panel ──────────────────────────────────────────────
    r2_colors = ["#ddd"] + ["#0A66C2" if r == max(all_r2s[1:]) else "#90b8d4"
                             for r in all_r2s[1:]]
    bars1 = ax1.bar(range(len(all_names)), all_r2s, color=r2_colors, edgecolor="none", width=0.55)
    ax1.set_xticks(range(len(all_names)))
    ax1.set_xticklabels(all_names, rotation=15, ha="right", fontsize=9)
    ax1.set_ylabel("R² score", fontsize=11)
    ax1.set_ylim(0, min(1.0, max(all_r2s) * 1.45))
    ax1.set_title("R²  — how much salary variation is explained?\n"
                  "0.0 = no better than guessing mean  |  1.0 = perfect",
                  fontsize=10, fontweight="bold")
    ax1.spines[["top", "right"]].set_visible(False)
    for bar, v in zip(bars1, all_r2s):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                 f"{v:.3f}\n({v*100:.0f}% explained)",
                 ha="center", va="bottom", fontsize=8, color="#333")

    # ── RMSE panel ────────────────────────────────────────────
    rmse_colors = ["#ddd"] + ["#0A66C2" if r == min(all_rmses[1:]) else "#90b8d4"
                               for r in all_rmses[1:]]
    bars2 = ax2.bar(range(len(all_names)), all_rmses, color=rmse_colors, edgecolor="none", width=0.55)
    ax2.set_xticks(range(len(all_names)))
    ax2.set_xticklabels(all_names, rotation=15, ha="right", fontsize=9)
    ax2.set_ylabel("RMSE — average prediction error", fontsize=11)
    ax2.set_title("RMSE  — average dollar error per salary prediction\n"
                  "Lower is better  |  grey bar = baseline to beat",
                  fontsize=10, fontweight="bold")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax2.spines[["top", "right"]].set_visible(False)
    for bar, v in zip(bars2, all_rmses):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + naive_rmse * 0.01,
                 f"${v/1000:.0f}K", ha="center", va="bottom", fontsize=9, color="#333")

    plt.suptitle("Salary prediction — model comparison", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "19_model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 9: REGRESSION — Predict Salary")
    print("=" * 60)

    df = load()
    X, y, feature_cols = prepare_features(df)

    if X is not None:
        results, best_name, X_test, y_test = train_and_evaluate(X, y, feature_cols)
        best_pred = results[best_name]["y_pred"]
        best_r2   = results[best_name]["r2"]
        rf_result = results.get("Random Forest", results[best_name])

        plot_actual_vs_predicted(y_test, best_pred, best_name, best_r2)
        plot_residuals(y_test, best_pred, best_name)
        plot_feature_importance_reg(rf_result["model"], feature_cols)
        plot_model_comparison(results, y_test)

    print("\nStep 9 complete. Proceed to: python 10_model_evaluation.py")
