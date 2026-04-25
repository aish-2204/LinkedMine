# =============================================================================
# STEP 3 — DATA PREPROCESSING
# Rubric: Data Preprocessing — missing values, encoding, outliers,
#         text cleaning, feature engineering, normalization
# =============================================================================

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import re
import nltk
import warnings
warnings.filterwarnings("ignore")

PROCESSED = "data/processed"
nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))

STATE_MAP = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
    'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
    'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DC',
}


# =============================================================================
# 1. LOAD MERGED DATA
# =============================================================================
def load_data():
    path = os.path.join(PROCESSED, "jobs_merged.csv")
    if not os.path.exists(path):
        print("Run 02_etl_warehouse.py first.")
        return None
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# =============================================================================
# 2. HANDLE MISSING VALUES
# =============================================================================
def handle_missing(df):
    print("\n--- Missing Value Treatment ---")
    before = df.isnull().sum().sum()

    # Salary: intentionally NOT imputed.
    # salary_clean is built from normalized_salary (real data only).
    # Imputing would fill 89% of rows with medians, making the target synthetic.

    # Categorical → 'Unknown'
    cat_cols = ["work_type", "formatted_experience_level", "company_size", "pay_period"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Numeric → 0  (applies nulls → 0; real applies start at 1, so 0 means no data)
    num_cols = ["applies", "views", "skill_count", "apply_rate"]
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    after = df.isnull().sum().sum()
    print(f"  Null count: {before:,} → {after:,}")
    print("  Note: salary nulls retained intentionally — salary_clean built from real data only")
    return df


# =============================================================================
# 3. NORMALIZE STATE COLUMN
# =============================================================================
def normalize_state(df):
    print("\n--- State Normalization ---")
    if "state" not in df.columns:
        return df

    df["state"] = df["state"].fillna("Unknown")
    df["state"] = df["state"].map(
        lambda x: STATE_MAP.get(str(x).strip(), str(x).strip())
    )
    # Keep only valid 2-letter abbreviations; everything else → Unknown
    df["state"] = df["state"].where(
        df["state"].str.match(r"^[A-Z]{2}$", na=False), other="Unknown"
    )
    known = (df["state"] != "Unknown").sum()
    print(f"  {known:,} rows with valid state  |  {df['state'].nunique()} unique values")
    return df


# =============================================================================
# 4. TEXT CLEANING (job descriptions)
# =============================================================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = text.lower()
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


def preprocess_text(df):
    print("\n--- Text Cleaning (job descriptions) ---")
    if "description" in df.columns:
        df["description_clean"] = df["description"].apply(clean_text)
        avg_len = df["description_clean"].str.split().str.len().mean()
        print(f"  Cleaned descriptions. Avg token count: {avg_len:.0f}")
    if "title" in df.columns:
        df["title_clean"] = df["title"].str.lower().str.strip()
    return df


# =============================================================================
# 5. ENCODE CATEGORICAL VARIABLES
# =============================================================================
def encode_categoricals(df):
    print("\n--- Categorical Encoding ---")

    exp_order = {
        "Internship": 0, "Entry level": 1, "Associate": 2,
        "Mid-Senior level": 3, "Director": 4, "Executive": 5, "Unknown": -1
    }
    if "formatted_experience_level" in df.columns:
        df["experience_encoded"] = (
            df["formatted_experience_level"].map(exp_order).fillna(-1).astype(int)
        )
        print("  experience_level → ordinal encoded (0–5)")

    if "work_type" in df.columns:
        dummies = pd.get_dummies(df["work_type"], prefix="work", drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        print(f"  work_type → {dummies.shape[1]} dummy columns")

    if "company_size" in df.columns:
        size_order = {
            "1-10": 1, "11-50": 2, "51-200": 3, "201-500": 4,
            "501-1000": 5, "1001-5000": 6, "5001-10000": 7, "10001+": 8, "Unknown": 0
        }
        df["company_size_encoded"] = (
            df["company_size"].map(size_order).fillna(0).astype(int)
        )
        print("  company_size → ordinal encoded (0–8)")

    return df


# =============================================================================
# 6. FEATURE ENGINEERING
# =============================================================================
def engineer_features(df):
    print("\n--- Feature Engineering ---")

    # Apply rate
    if {"applies", "views"}.issubset(df.columns):
        df["apply_rate"] = (df["applies"] / df["views"].replace(0, 1)).round(4)
        print("  Created: apply_rate = applies / views")

    # salary_clean: real salary data only, sourced from normalized_salary in postings.csv.
    # Avoids the corrupted salary_midpoint which was 89% imputed from a partial join.
    if "normalized_salary" in df.columns:
        df["salary_clean"] = df["normalized_salary"].where(
            df["normalized_salary"].between(20_000, 500_000)
        )
        non_null = df["salary_clean"].notna().sum()
        print(f"  Created: salary_clean — {non_null:,} rows with real salary ($20K–$500K)")

    # high_salary: top 25% of known salaries (NaN salary rows default to 0)
    if "salary_clean" in df.columns:
        p75 = df["salary_clean"].quantile(0.75)
        df["high_salary"] = (df["salary_clean"] >= p75).astype(int)
        print(f"  Created: high_salary (1 if salary_clean >= ${p75:,.0f})")

    # high_demand: threshold computed only on rows with real applies data (applies > 0).
    # Applies nulls were filled to 0; real applies start at 1 in the raw data.
    # Rows with applies == 0 receive NaN and are excluded from classification.
    if "applies" in df.columns:
        valid     = df.loc[df["applies"] > 0, "applies"]
        p75_app   = valid.quantile(0.75)
        df["high_demand"] = np.where(
            df["applies"] > 0,
            (df["applies"] >= p75_app).astype(float),
            np.nan
        )
        known = (df["applies"] > 0).sum()
        high  = (df["high_demand"] == 1).sum()
        print(f"  Created: high_demand — threshold={p75_app:.0f} applies "
              f"| {known:,} rows with data, {high:,} high-demand ({high/known*100:.1f}%)")

    # Title seniority flag
    if "title" in df.columns:
        senior_kw = ["senior", "lead", "principal", "staff", "director", "vp", "head", "chief"]
        df["is_senior_title"] = df["title"].str.lower().str.contains(
            "|".join(senior_kw), na=False
        ).astype(int)
        print("  Created: is_senior_title")

    return df


# =============================================================================
# 7. SCALE NUMERIC FEATURES
# =============================================================================
def scale_features(df):
    print("\n--- Feature Scaling (StandardScaler) ---")
    scale_cols = [c for c in ["salary_clean", "applies", "views", "skill_count", "apply_rate"]
                  if c in df.columns]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[scale_cols].fillna(0))
    scaled_df = pd.DataFrame(
        scaled, columns=[f"{c}_scaled" for c in scale_cols], index=df.index
    )
    df = pd.concat([df, scaled_df], axis=1)
    print(f"  Scaled: {scale_cols}")
    return df, scaler


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("STEP 3: DATA PREPROCESSING")
    print("=" * 60)

    df = load_data()
    if df is None:
        exit()

    df = handle_missing(df)
    df = preprocess_text(df)
    df = encode_categoricals(df)
    df = normalize_state(df)
    df = engineer_features(df)
    df, scaler = scale_features(df)

    out_path = os.path.join(PROCESSED, "jobs_preprocessed.csv")
    df.to_csv(out_path, index=False)

    print(f"\nFinal dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  salary_clean non-null : {df['salary_clean'].notna().sum():,}")
    print(f"  high_demand non-null  : {df['high_demand'].notna().sum():,}")
    print(f"Saved → {out_path}")
    print("\nStep 3 complete. Proceed to: python 04_visualization.py")
