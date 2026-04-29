# Mining the Job Market
## Discovering Skill Demand, Salary Patterns & Hiring Trends from 126,000 LinkedIn Postings

---

## Project Overview

A full data mining pipeline applied to a real-world LinkedIn jobs dataset.  
The project covers every stage from raw data ingestion to model evaluation, producing 20+ figures and trained models.

**Key questions answered:**
- Which job categories are most in demand and where?
- What salary should you expect at each experience level?
- Which job categories consistently appear together in postings?
- Can we predict whether a job will be high-demand or forecast its salary?

---

## Pipeline (10 Steps)

| Step | File | What it does |
|------|------|-------------|
| 1 | `01_data_collection.py` | Verify raw files, profile source CSVs |
| 2 | `02_etl_warehouse.py` | ETL pipeline → SQLite data warehouse |
| 3 | `03_preprocessing.py` | Clean data, engineer features, encode variables |
| 4 | `04_eda.py` | Exploratory Data Analysis — distributions, missing values |
| 5 | `05_visualization.py` | 8 charts: bar, boxplot, scatter, heatmap, CDF, line |
| 6 | `06_mining_classification.py` | Predict high-demand jobs (Random Forest + Gradient Boosting) |
| 7 | `07_mining_clustering.py` | K-Means clustering — discover job role families |
| 8 | `08_mining_association_rules.py` | Apriori — find job categories that co-occur |
| 9 | `09_mining_regression.py` | Predict annual salary (regression models) |
| 10 | `10_model_evaluation.py` | Consolidated evaluation report for all models |

---

## Project Structure

```
linkedin_jobs_project/
│
├── data/
│   ├── raw/                  # Downloaded CSVs from Kaggle (place files here)
│   ├── processed/            # Cleaned, merged dataset
│   └── warehouse/            # SQLite database (etl_warehouse.db)
│
├── outputs/
│   └── figures/              # All saved plots (20+ figures)
│
├── 01_data_collection.py
├── 02_etl_warehouse.py
├── 03_preprocessing.py
├── 04_eda.py
├── 05_visualization.py
├── 06_mining_classification.py
├── 07_mining_clustering.py
├── 08_mining_association_rules.py
├── 09_mining_regression.py
├── 10_model_evaluation.py
├── requirements.txt
└── README.md
```

---

## Dataset Download

1. Go to: https://www.kaggle.com/datasets/arshkon/linkedin-job-postings
2. Download and unzip into `data/raw/` — keep the internal sub-folders as-is

**Dataset stats after processing:** 123,849 job postings · 63 columns · 126,000 skill records

---

## Run Order

```bash
pip install -r requirements.txt

python 01_data_collection.py       # Verify & profile raw files
python 02_etl_warehouse.py         # Build SQLite warehouse (ETL)
python 03_preprocessing.py         # Clean & engineer features
python 04_eda.py                   # Exploratory data analysis
python 05_visualization.py         # Generate all charts
python 06_mining_classification.py # Classification model
python 07_mining_clustering.py     # K-Means clustering
python 08_mining_association_rules.py  # Apriori association rules
python 09_mining_regression.py     # Salary regression model
python 10_model_evaluation.py      # Consolidated evaluation report
```

---

## Output Figures

| Figure | Description |
|--------|-------------|
| 01 | Top 15 job titles by posting volume |
| 02 | Top 15 job categories on LinkedIn |
| 03 | Salary distribution by experience level (boxplot) |
| 04 | Salary vs applicants scatter (do higher-paying jobs attract more?) |
| 05 | Feature correlation heatmap |
| 06 | Salary cumulative distribution (CDF) |
| 07 | Top 15 states by posting volume |
| 08 | Remote vs on-site breakdown |
| 09 | EDA — salary distribution + category tags per posting |
| 10–11 | Classification: feature importance + ROC curve |
| 12 | K-Means cluster scatter (PCA) — job role families |
| 13 | Association rules — labeled bubble chart (Support vs Confidence) |
| 14 | Association rules — 3-panel metric chart (Support, Confidence, Lift) |
| 15 | Job category co-occurrence heatmap |
| 16–19 | Regression: predicted vs actual, residuals, feature importance |
| 20 | Model evaluation scorecard (all 4 mining tasks) |

---

## Key Findings

- **Salary:** Median $82,500 · Mean $96K · only 29% of postings disclose salary
- **Top categories:** Management, Sales, IT, Engineering dominate posting volume
- **Clustering:** 5 distinct job role families found (K=5, silhouette = 0.44)
- **Association rules:** Strongest pairs — Management↔Manufacturing (Lift 5.4×), Sales↔Business Dev (Lift 5.2×), IT↔Engineering (Lift 3.4×)
- **High-demand classification:** ROC-AUC > 0.85
- **Salary regression:** R² explained variance reported in Step 10

---

## Future Work

1. **NLP on job descriptions** — extract actual skills from free text (BERT / LDA) to replace sparse category tags
2. **Salary imputation** — predict salary for the 71% of jobs that don't disclose it
3. **Temporal forecasting** — track skill demand shifts over time with a time-series dataset
4. **Geographic analysis** — map salary gaps and skill demand by state/city
5. **Job-candidate matching** — use cluster output as a recommendation engine
6. **Fairness audit** — investigate salary disparities by remote status, company size, and location
