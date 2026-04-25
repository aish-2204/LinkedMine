# =============================================================================
# STEP 1 — DATA COLLECTION & SOURCE IDENTIFICATION
# Rubric: Problem Definition, Data Source Identification, Data Collection
# =============================================================================
# Dataset: LinkedIn Job Postings 2023
# Source:  https://www.kaggle.com/datasets/arshkon/linkedin-job-postings
# Method:  CSV download (Kaggle public repository)
# Domain:  Jobs / HR / Labor Economics
# =============================================================================

import os
import pandas as pd
import numpy as np

RAW_DIR = "data/raw"

# -----------------------------------------------------------------------------
# Expected files after Kaggle download + unzip
# -----------------------------------------------------------------------------
EXPECTED_FILES = {
    "job_postings":   "postings.csv",
    "companies":      "companies/companies.csv",
    "job_skills":     "jobs/job_skills.csv",
    "job_industries": "jobs/job_industries.csv",
    "salaries":       "jobs/salaries.csv",
    "benefits":       "jobs/benefits.csv",
}


def check_files():
    """Verify all expected raw files are present."""
    print("=" * 60)
    print("STEP 1: Data Collection — File Check")
    print("=" * 60)
    missing = []
    for name, fname in EXPECTED_FILES.items():
        path = os.path.join(RAW_DIR, fname)
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {name:20s} → {fname}")
        if not exists:
            missing.append(fname)

    if missing:
        print(f"\n  Download from Kaggle and place in {RAW_DIR}/")
        print("  https://www.kaggle.com/datasets/arshkon/linkedin-job-postings")
        return False
    print("\n  All files present.\n")
    return True


def load_all():
    """Load all CSVs into a dict of DataFrames."""
    dfs = {}
    for name, fname in EXPECTED_FILES.items():
        path = os.path.join(RAW_DIR, fname)
        if os.path.exists(path):
            dfs[name] = pd.read_csv(path, low_memory=False)
    return dfs


def profile_dataset(dfs):
    """Print a data profile for each table — rubric: problem definition."""
    print("=" * 60)
    print("DATA PROFILE — Shape, Nulls, Types")
    print("=" * 60)

    for name, df in dfs.items():
        print(f"\n--- {name.upper()} ---")
        print(f"  Rows: {len(df):,}  |  Columns: {df.shape[1]}")
        print(f"  Columns: {list(df.columns)}")

        null_pct = (df.isnull().sum() / len(df) * 100).round(1)
        high_null = null_pct[null_pct > 10]
        if not high_null.empty:
            print("  Columns with >10% nulls:")
            for col, pct in high_null.items():
                print(f"    {col}: {pct}% missing")

    # Research questions summary
    print("\n" + "=" * 60)
    print("RESEARCH QUESTIONS (Problem Definition)")
    print("=" * 60)
    questions = [
        "1. Which skills appear together in job postings? (Association Rules)",
        "2. What features predict salary? (Regression)",
        "3. What job role families exist based on skill profiles? (Clustering)",
        "4. Can we predict high-demand job postings from posting attributes? (Classification)",
    ]
    for q in questions:
        print(f"  {q}")

    print("\nTarget Variables:")
    print("  Regression     → salary_midpoint  (continuous USD)")
    print("  Classification → high_demand       (binary: applies >= 75th percentile)")


def describe_sources():
    """Document data source provenance for the rubric."""
    print("\n" + "=" * 60)
    print("DATA SOURCE DOCUMENTATION")
    print("=" * 60)
    sources = {
        "Repository":    "Kaggle — Public Dataset Repository",
        "Origin":        "Scraped from LinkedIn in 2023",
        "License":       "CC0 (Public Domain)",
        "Collection":    "CSV download — no API key required",
        "Format":        "6 relational CSV files",
        "Total records": "~33,000 job postings + related tables",
        "URL":           "kaggle.com/datasets/arshkon/linkedin-job-postings",
    }
    for k, v in sources.items():
        print(f"  {k:20s}: {v}")


if __name__ == "__main__":
    files_ok = check_files()
    if files_ok:
        dfs = load_all()
        profile_dataset(dfs)
    describe_sources()
    print("\nStep 1 complete. Proceed to: python 02_etl_warehouse.py")
