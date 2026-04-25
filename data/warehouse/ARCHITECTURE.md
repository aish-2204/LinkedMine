# Star Schema — LinkedIn Jobs Data Warehouse

## Why a Star Schema?

The raw Kaggle dataset arrives as 6 separate CSV files. Joining them ad-hoc
in every analysis script is fragile and slow. A **star schema** pre-integrates
them into one fact table surrounded by dimension tables, making analytical
queries fast and repeatable.

---

## Schema Diagram

```
                       ┌─────────────────────┐
                       │    dim_companies     │
                       │─────────────────────│
                       │ company_id  (PK)     │
                       │ name                 │
                       │ company_size         │
                       │ state / country      │
                       └──────────┬──────────┘
                                  │ company_id (FK)
                                  │
         ┌────────────────────────▼──────────────────────────┐
         │                fact_job_postings                   │
         │──────────────────────────────────────────────────  │
         │  job_id            (PK)                            │
         │  company_id        (FK → dim_companies)            │
         │  title / work_type / formatted_experience_level    │
         │  is_remote  /  remote_allowed                      │
         │  applies  /  views  /  apply_rate                  │
         │  skill_count  /  listed_date  /  listed_month      │
         │  normalized_salary  /  salary_clean  (derived)     │
         │  high_demand  /  high_salary  /  is_senior_title   │
         └───────┬───────────────────────────┬───────────────┘
                 │ job_id (FK)               │ job_id (FK)
                 │                           │
    ┌────────────▼──────────┐   ┌────────────▼──────────────┐
    │    dim_salaries        │   │    bridge_job_skills       │
    │───────────────────────│   │───────────────────────────│
    │  job_id    (FK/PK)    │   │  job_id     (FK)           │
    │  min_salary            │   │  skill_abr  (category code)│
    │  max_salary            │   │                            │
    │  salary_midpoint       │   │  One row per job-category  │
    │  pay_period / currency │   │  pairing. A job can belong │
    └───────────────────────┘   │  to 1–3 categories.        │
                                └────────────────────────────┘
                                         │ job_id (FK)
                    ┌────────────────────▼──────────────┐
                    │           dim_benefits             │
                    │──────────────────────────────────  │
                    │  job_id   (FK)                     │
                    │  type  (Medical, Dental, 401K …)   │
                    │  inferred  (True/False)             │
                    └───────────────────────────────────┘
```

---

## Table Descriptions

### `fact_job_postings`  *(central fact table)*
Every row is one LinkedIn job posting. This is the analytical hub — all
dimension and bridge tables join back to it via `job_id`.

| Column | Type | Notes |
|--------|------|-------|
| `job_id` | INT | Primary key |
| `company_id` | INT | FK → dim_companies |
| `title` | TEXT | Raw job title |
| `work_type` | TEXT | FULL_TIME / CONTRACT / PART_TIME … |
| `formatted_experience_level` | TEXT | Entry level / Mid-Senior / Director … |
| `is_remote` | INT | 1 = remote, 0 = on-site (derived from remote_allowed) |
| `applies` | FLOAT | Number of applicants (null-filled to 0) |
| `views` | FLOAT | Number of views |
| `apply_rate` | FLOAT | applies / views |
| `skill_count` | INT | Number of skill categories listed |
| `listed_date` | DATETIME | Parsed from original_listed_time (ms) |
| `normalized_salary` | FLOAT | Annual salary from postings.csv (most reliable source) |
| `salary_clean` | FLOAT | normalized_salary capped $20K–$500K (used in regression) |
| `high_demand` | FLOAT | 1 if applies ≥ 8 (p75); NaN if applies unknown |
| `high_salary` | INT | 1 if salary_clean ≥ p75 ($125K) |
| `is_senior_title` | INT | 1 if title contains senior/lead/director … |

---

### `dim_companies`  *(company dimension)*
One row per company. Joined to the fact table on `company_id`.

| Column | Notes |
|--------|-------|
| `company_id` | Primary key |
| `name` | Company name |
| `company_size` | Ordinal band: 1–10 / 11–50 / … / 10001+ |
| `state` | Normalised 2-letter US state code |
| `country` | Country code |

---

### `dim_salaries`  *(salary dimension — partial coverage)*
Salary ranges from a separate `salaries.csv` file. Only **12,750 of 123,849**
jobs have a match here (~10%). This is why `normalized_salary` from
`fact_job_postings` is preferred for analysis.

| Column | Notes |
|--------|-------|
| `job_id` | FK → fact_job_postings |
| `min_salary` / `max_salary` | Raw salary range |
| `salary_midpoint` | (min + max) / 2 |
| `pay_period` | YEARLY / HOURLY / MONTHLY |

---

### `bridge_job_skills`  *(many-to-many bridge)*
Each job can belong to 1–3 industry categories. This is a bridge table
resolving the many-to-many relationship between jobs and categories.

| Column | Notes |
|--------|-------|
| `job_id` | FK → fact_job_postings |
| `skill_abr` | Industry code: IT, SALE, MGMT, HCPR, ENG … (35 unique values) |

> **Note:** Despite the column name `skill_abr`, these are **industry/function
> category codes**, not individual skills. The dataset uses abbreviated
> occupational categories rather than specific tool names.

---

### `dim_benefits`  *(benefit types per job)*
One row per job-benefit pairing. A single job can have multiple benefit rows.

| Column | Notes |
|--------|-------|
| `job_id` | FK → fact_job_postings |
| `type` | Medical insurance / Dental / 401K / Remote work … |
| `inferred` | True if the benefit was inferred by the scraper |

---

## ETL Pipeline

```
  Raw CSVs (data/raw/)
       │
       ▼  EXTRACT  (02_etl_warehouse.py)
  Load 6 CSVs into pandas DataFrames
       │
       ▼  TRANSFORM
  • Standardise column names (lowercase, underscores)
  • Parse timestamps (original_listed_time ms → datetime)
  • Compute salary_midpoint = (min + max) / 2
  • Derive is_remote from remote_allowed
  • Compute skill_count per job_id
  • Compute apply_rate = applies / views
  • Merge companies + salaries into fact table
       │
       ▼  LOAD
  Write to SQLite: data/warehouse/etl_warehouse.db
  Write merged CSV: data/processed/jobs_merged.csv
  Write skills CSV: data/processed/job_skills_clean.csv
       │
       ▼  PREPROCESS  (03_preprocessing.py)
  • Missing value treatment (salary intentionally not imputed)
  • State column normalisation (full name → 2-letter code)
  • Text cleaning (job descriptions — NLTK stopword removal)
  • Categorical encoding (experience level ordinal, work_type OHE)
  • Feature engineering (salary_clean, high_salary, high_demand,
    is_senior_title, apply_rate)
  • StandardScaler on numeric features
  Save → data/processed/jobs_preprocessed.csv
```

---

## Design Decisions

| Decision | Reason |
|----------|--------|
| SQLite over CSV | Enables SQL queries, indexing, and reproducible joins |
| Star schema over flat file | Avoids repeating company data on every job row |
| `normalized_salary` preferred over `salary_midpoint` | salaries.csv only covers 10% of jobs; normalized_salary is in postings.csv for 29% of jobs |
| salary NOT imputed | Imputing 89% of salary values with medians produces a synthetic target that corrupts regression and salary visualisations |
| `bridge_job_skills` as separate table | A job has 1–3 categories — storing as a comma list in the fact table would violate 1NF and complicate Apriori input preparation |
