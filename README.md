# Mining the Job Market
## Discovering Skill Demand, Salary Patterns & Hiring Trends from 33,000 LinkedIn Postings

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
│   └── figures/              # Saved plots
│
├── 01_data_collection.py
├── 02_etl_warehouse.py
├── 03_preprocessing.py
├── 04_visualization.py
├── 05_mining_classification.py
├── 06_mining_clustering.py
├── 07_mining_association_rules.py
├── 08_mining_regression.py
├── 09_model_evaluation.py
├── requirements.txt
└── README.md
```

## Dataset Download
1. Go to: https://www.kaggle.com/datasets/arshkon/linkedin-job-postings
2. Download and unzip into `data/raw/`
3. You should have: `job_postings.csv`, `companies/companies.csv`, `job_skills.csv`,
   `job_industries.csv`, `salaries.csv`, `benefits.csv`

## Run Order
```bash
pip install -r requirements.txt
python 01_data_collection.py      # Verify & profile raw data
python 02_etl_warehouse.py        # Build SQLite warehouse
python 03_preprocessing.py        # Clean & engineer features
python 04_visualization.py        # Generate all plots
python 05_mining_classification.py
python 06_mining_clustering.py
python 07_mining_association_rules.py
python 08_mining_regression.py
python 09_model_evaluation.py     # Consolidated evaluation report
```
