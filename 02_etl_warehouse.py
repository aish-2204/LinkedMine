# =============================================================================
# STEP 2 — DATA WAREHOUSING: ETL PIPELINE
# Rubric: Data Warehousing — Extract, Transform, Load, Data Integration
# =============================================================================
# Builds a SQLite star schema from 6 raw CSV files.
#
# Star Schema:
#   FACT:   job_postings  (job_id, company_id, salary_id, views, applies, ...)
#   DIM:    companies     (company_id, name, size, industry, location)
#   DIM:    salaries      (salary_id, min_salary, max_salary, currency, type)
#   BRIDGE: job_skills    (job_id, skill_abr)
#   BRIDGE: job_industries(job_id, industry_id)
#   DIM:    benefits      (job_id, type, inferred)
# =============================================================================

import os
import pandas as pd
import sqlite3

RAW_DIR    = "data/raw"
DB_PATH    = "data/warehouse/etl_warehouse.db"
PROCESSED  = "data/processed"

os.makedirs("data/warehouse", exist_ok=True)
os.makedirs(PROCESSED, exist_ok=True)


# =============================================================================
# EXTRACT — Load raw CSVs
# =============================================================================
def extract():
    print("=" * 60)
    print("ETL STEP 1: EXTRACT")
    print("=" * 60)

    dfs = {}
    files = {
        "job_postings":   "postings.csv",
        "companies":      "companies/companies.csv",
        "job_skills":     "jobs/job_skills.csv",
        "job_industries": "jobs/job_industries.csv",
        "salaries":       "jobs/salaries.csv",
        "benefits":       "jobs/benefits.csv",
    }
    for name, fname in files.items():
        path = os.path.join(RAW_DIR, fname)
        if os.path.exists(path):
            dfs[name] = pd.read_csv(path, low_memory=False)
            print(f"  Loaded {name:20s}: {len(dfs[name]):>7,} rows")
        else:
            print(f"  SKIP   {name:20s}: file not found — using sample data")
            dfs[name] = _generate_sample(name)

    return dfs


# =============================================================================
# TRANSFORM — Clean and integrate
# =============================================================================
def transform(dfs):
    print("\n" + "=" * 60)
    print("ETL STEP 2: TRANSFORM")
    print("=" * 60)

    jobs = dfs["job_postings"].copy()

    # --- Standardise column names ---
    jobs.columns = jobs.columns.str.strip().str.lower().str.replace(" ", "_")

    # --- Parse timestamps ---
    if "original_listed_time" in jobs.columns:
        jobs["listed_date"] = pd.to_datetime(jobs["original_listed_time"], unit="ms", errors="coerce")
        jobs["listed_month"] = jobs["listed_date"].dt.to_period("M").astype(str)

    # --- Salary midpoint ---
    salaries = dfs["salaries"].copy()
    salaries.columns = salaries.columns.str.strip().str.lower().str.replace(" ", "_")
    if {"min_salary", "max_salary"}.issubset(salaries.columns):
        salaries["salary_midpoint"] = (
            salaries["min_salary"].fillna(salaries["max_salary"]) +
            salaries["max_salary"].fillna(salaries["min_salary"])
        ) / 2
        # Remove extreme outliers (keep salaries between $10K and $500K annual)
        mask = salaries["salary_midpoint"].between(10_000, 500_000)
        salaries_clean = salaries[mask].copy()
        print(f"  Salary rows after outlier removal: {len(salaries_clean):,} / {len(salaries):,}")
    else:
        salaries_clean = salaries

    # --- Remote flag ---
    if "remote_allowed" in jobs.columns:
        jobs["is_remote"] = jobs["remote_allowed"].fillna(0).astype(int)
    else:
        jobs["is_remote"] = 0

    # --- Companies ---
    companies = dfs["companies"].copy()
    companies.columns = companies.columns.str.strip().str.lower().str.replace(" ", "_")

    # --- Merge fact with dimensions ---
    if "company_id" in jobs.columns and "company_id" in companies.columns:
        jobs_merged = jobs.merge(
            companies[["company_id", "name", "company_size", "state", "country"]],
            on="company_id", how="left"
        )
    else:
        jobs_merged = jobs.copy()

    if "job_id" in jobs_merged.columns and "job_id" in salaries_clean.columns:
        jobs_merged = jobs_merged.merge(
            salaries_clean[["job_id", "salary_midpoint", "min_salary", "max_salary", "pay_period"]],
            on="job_id", how="left"
        )

    print(f"  Merged dataset shape: {jobs_merged.shape}")
    print(f"  Columns: {list(jobs_merged.columns)}")

    # --- Skills bridge ---
    skills = dfs["job_skills"].copy()
    skills.columns = skills.columns.str.strip().str.lower().str.replace(" ", "_")

    # Skill count per job (feature engineering)
    if "job_id" in skills.columns:
        skill_counts = skills.groupby("job_id").size().reset_index(name="skill_count")
        jobs_merged = jobs_merged.merge(skill_counts, on="job_id", how="left")
        jobs_merged["skill_count"] = jobs_merged["skill_count"].fillna(0).astype(int)

    # --- Apply rate (feature engineering) ---
    if {"applies", "views"}.issubset(jobs_merged.columns):
        jobs_merged["apply_rate"] = (
            jobs_merged["applies"] / jobs_merged["views"].replace(0, pd.NA)
        ).round(4)

    print(f"  Feature engineered columns added: skill_count, apply_rate, is_remote, listed_month")

    return jobs_merged, skills, salaries_clean, companies


# =============================================================================
# LOAD — Write to SQLite warehouse
# =============================================================================
def load(jobs_merged, skills, salaries, companies, dfs):
    print("\n" + "=" * 60)
    print("ETL STEP 3: LOAD → SQLite Warehouse")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)

    tables = {
        "fact_job_postings": jobs_merged,
        "dim_companies":     companies,
        "dim_salaries":      salaries,
        "bridge_job_skills": skills,
        "dim_benefits":      dfs["benefits"],
    }

    for table_name, df in tables.items():
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"  Loaded {table_name:30s}: {len(df):>7,} rows")

    # Save merged CSV for downstream scripts
    out_path = os.path.join(PROCESSED, "jobs_merged.csv")
    jobs_merged.to_csv(out_path, index=False)
    print(f"\n  Merged CSV saved → {out_path}")

    # Save skills for association rules
    skills_path = os.path.join(PROCESSED, "job_skills_clean.csv")
    skills.to_csv(skills_path, index=False)

    conn.close()
    print(f"\n  SQLite warehouse → {DB_PATH}")
    print("  Schema: fact_job_postings + dim_companies + dim_salaries + bridge_job_skills + dim_benefits")


# =============================================================================
# SAMPLE DATA GENERATOR (fallback when Kaggle files not yet downloaded)
# =============================================================================
def _generate_sample(name):
    """Generate minimal sample data for testing the pipeline."""
    import numpy as np
    np.random.seed(42)
    n = 500

    if name == "job_postings":
        return pd.DataFrame({
            "job_id":           range(1, n+1),
            "company_id":       np.random.randint(1, 100, n),
            "title":            np.random.choice(["Data Scientist","ML Engineer","Data Analyst","Software Engineer","Product Manager"], n),
            "description":      ["Sample job description " * 5] * n,
            "work_type":        np.random.choice(["FULL_TIME","PART_TIME","CONTRACT"], n),
            "remote_allowed":   np.random.choice([0, 1], n, p=[0.6, 0.4]),
            "formatted_experience_level": np.random.choice(["Entry level","Mid-Senior level","Director","Associate"], n),
            "applies":          np.random.randint(0, 500, n),
            "views":            np.random.randint(100, 5000, n),
            "original_listed_time": pd.date_range("2023-01-01", periods=n, freq="h").astype(int) // 10**6,
        })
    elif name == "companies":
        return pd.DataFrame({
            "company_id":    range(1, 101),
            "name":          [f"Company {i}" for i in range(1, 101)],
            "company_size":  np.random.choice(["1-10","11-50","51-200","201-500","501-1000","1001-5000","5001-10000","10001+"], 100),
            "state":         np.random.choice(["CA","NY","TX","WA","MA"], 100),
            "country":       "US",
        })
    elif name == "job_skills":
        skills_pool = ["Python","SQL","Machine Learning","Excel","Communication","Leadership",
                       "Java","AWS","Tableau","R","TensorFlow","Data Analysis","Project Management","Azure"]
        rows = []
        for job_id in range(1, n+1):
            for skill in np.random.choice(skills_pool, np.random.randint(2, 7), replace=False):
                rows.append({"job_id": job_id, "skill_abr": skill})
        return pd.DataFrame(rows)
    elif name == "salaries":
        min_s = np.random.randint(40000, 120000, n)
        return pd.DataFrame({
            "job_id":      range(1, n+1),
            "min_salary":  min_s,
            "max_salary":  min_s + np.random.randint(10000, 60000, n),
            "pay_period":  "YEARLY",
            "currency":    "USD",
        })
    elif name == "job_industries":
        return pd.DataFrame({
            "job_id":      np.random.choice(range(1, n+1), n),
            "industry_id": np.random.randint(1, 30, n),
        })
    elif name == "benefits":
        benefit_types = ["Medical insurance","Dental insurance","401K","Remote work","Vision insurance"]
        rows = []
        for job_id in range(1, n+1):
            for b in np.random.choice(benefit_types, np.random.randint(1, 4), replace=False):
                rows.append({"job_id": job_id, "type": b, "inferred": np.random.choice([True, False])})
        return pd.DataFrame(rows)
    return pd.DataFrame()


if __name__ == "__main__":
    dfs                                    = extract()
    jobs_merged, skills, salaries, companies = transform(dfs)
    load(jobs_merged, skills, salaries, companies, dfs)
    print("\nStep 2 complete. Proceed to: python 03_preprocessing.py")
