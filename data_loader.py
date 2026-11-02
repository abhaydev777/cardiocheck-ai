"""
Data loader and generator for Heart Disease Classification.
Matches the authentic UCI Heart Disease (Cleveland Clinic) schema:
  - age: age in years
  - sex: 1 = male, 0 = female
  - cp: chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)
  - trestbps: resting blood pressure (in mm Hg on admission to hospital)
  - chol: serum cholesterol in mg/dl
  - fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
  - restecg: resting electrocardiographic results (0 = normal, 1 = ST-T wave abnormality, 2 = LV hypertrophy)
  - thalach: maximum heart rate achieved
  - exang: exercise induced angina (1 = yes; 0 = no)
  - oldpeak: ST depression induced by exercise relative to rest
  - target: 0 = Healthy (absence of heart disease), 1 = Heart Disease (> 50% diameter narrowing)
"""

import os
import urllib.request
import numpy as np
import pandas as pd

CSV_PATH = "heart.csv"

COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "target",
]

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"


def generate_realistic_dataset(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Clinically calibrated synthetic cardiovascular dataset matching
    Framingham & Cleveland clinical distributions.
    """
    rng = np.random.default_rng(random_state)

    age = rng.integers(29, 78, size=n_samples)
    sex = rng.binomial(1, 0.68, size=n_samples)  # 1 = male, 0 = female

    # cp: 0=typical angina, 1=atypical angina, 2=non-anginal, 3=asymptomatic
    cp = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.10, 0.20, 0.30, 0.40])

    trestbps = np.clip(np.round(rng.normal(131.6, 17.5, size=n_samples) + (age - 54) * 0.3), 94, 200).astype(int)
    chol = np.clip(np.round(rng.normal(246.0, 51.0, size=n_samples) + (age - 54) * 0.4), 126, 560).astype(int)
    fbs = rng.binomial(1, 0.15, size=n_samples)
    restecg = rng.choice([0, 1, 2], size=n_samples, p=[0.49, 0.48, 0.03])

    base_thalach = 208 - (0.7 * age)
    thalach = np.clip(np.round(rng.normal(base_thalach, 18.0)), 71, 205).astype(int)

    exang_prob = np.clip(0.15 + 0.004 * (age - 40) + 0.25 * (cp == 3), 0.05, 0.85)
    exang = rng.binomial(1, exang_prob)

    oldpeak_raw = rng.exponential(scale=0.8, size=n_samples) + 0.6 * exang + 0.4 * (cp == 3)
    oldpeak = np.clip(np.round(oldpeak_raw, 1), 0.0, 6.2)

    # Risk log-odds calibrated: higher oldpeak, exang, cp=3 (asymptomatic CAD), age, sex=1 -> target=1
    z = (
        -2.5
        + 0.035 * (age - 54)
        + 0.65 * sex
        + 0.60 * (cp == 3)  # Silent ischemia / asymptomatic in stress cohort
        + 0.012 * (trestbps - 130)
        + 0.003 * (chol - 240)
        + 0.25 * fbs
        + 0.30 * (restecg > 0)
        - 0.025 * (thalach - 150)
        + 1.10 * exang
        + 0.85 * oldpeak
    )
    prob = 1.0 / (1.0 + np.exp(-z))
    target = rng.binomial(1, prob)

    df = pd.DataFrame({
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "target": target,
    })
    return df


def download_uci_dataset() -> pd.DataFrame:
    """
    Downloads and parses the authentic UCI Cleveland Clinic dataset.
    """
    try:
        req = urllib.request.Request(UCI_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = resp.read().decode("utf-8")

        lines = [l.strip().split(",") for l in data.strip().split("\n") if l.strip()]
        raw_cols = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
        ]
        df = pd.DataFrame(lines, columns=raw_cols)
        # Drop missing values represented by '?'
        df = df.replace("?", np.nan).dropna()

        # Cast numeric fields
        for col in raw_cols:
            df[col] = pd.to_numeric(df[col])

        # Standardize target: 0 = Healthy, 1 = Heart Disease (>0)
        df["target"] = (df["num"] > 0).astype(int)

        # Standardize cp to 0-indexed (0: Typical, 1: Atypical, 2: Non-Anginal, 3: Asymptomatic)
        # In original UCI: 1, 2, 3, 4 -> remap to 0, 1, 2, 3
        df["cp"] = (df["cp"] - 1).astype(int)

        df = df[COLUMNS].copy()
        print(f"[INFO] Successfully downloaded authentic UCI Heart Disease data ({len(df)} samples)")
        return df
    except Exception as e:
        print(f"[WARN] Failed to download from official UCI mirror ({e})")
        return None


def get_dataset(force_refresh: bool = False) -> pd.DataFrame:
    """
    Returns the heart disease dataset.
    Loads from local heart.csv if present, else fetches authentic UCI data or generates realistic data.
    """
    if os.path.exists(CSV_PATH) and not force_refresh:
        df = pd.read_csv(CSV_PATH)
        if all(c in df.columns for c in COLUMNS):
            print(f"[INFO] Loaded existing dataset from {CSV_PATH} ({len(df)} samples)")
            return df[COLUMNS]

    df = download_uci_dataset()

    if df is None or len(df) < 100:
        print("[INFO] Generating clinically calibrated synthetic dataset matching UCI schema...")
        df = generate_realistic_dataset(n_samples=600, random_state=42)

    df = df.astype({
        "age": int,
        "sex": int,
        "cp": int,
        "trestbps": int,
        "chol": int,
        "fbs": int,
        "restecg": int,
        "thalach": int,
        "exang": int,
        "oldpeak": float,
        "target": int,
    })

    df.to_csv(CSV_PATH, index=False)
    print(f"[SUCCESS] Dataset saved to {CSV_PATH} with {len(df)} rows.")
    return df


if __name__ == "__main__":
    df = get_dataset(force_refresh=True)
    print("\nDataset Preview:")
    print(df.head())
    print("\nClass Distribution:")
    print(df["target"].value_counts(normalize=True).rename({0: "Healthy (0)", 1: "Disease (1)"}))
    print("\nFeature Correlations with Target (0=Healthy, 1=Heart Disease):")
    print(df.corr()["target"].sort_values(ascending=False))
