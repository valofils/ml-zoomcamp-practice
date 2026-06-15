# =============================================================================
# MODULE 01 — Introduction to Machine Learning
# Dataset : WFP Global Food Prices Database (HDX)
# Source  : https://data.humdata.org/dataset/wfp-food-prices
# Task    : Understand the data and frame the ML problem
# =============================================================================

# -----------------------------------------------------------------------------
# WHAT IS MACHINE LEARNING?
# Rule-based systems require a human to hard-code every decision.
# ML systems learn the decision rules directly from data.
#
# Example in this context:
#   Rule-based : IF country == "Ethiopia" AND commodity == "Maize" → price range X
#   ML-based   : learn price patterns from thousands of market observations
#                and generalize to unseen markets or future months
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# SUPERVISED LEARNING SETUP
#   - Feature matrix X  : country, market, commodity, currency, month, year, ...
#   - Target vector  y  : price (USD)
#   - Model          g  : g(X) ≈ y
#   - Loss function     : measures how far g(X) is from y (e.g. RMSE)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# CRISP-DM FRAMEWORK (Cross-Industry Standard Process for Data Mining)
#   1. Business Understanding  — why do we want to predict food prices?
#   2. Data Understanding      — what does the WFP dataset contain?
#   3. Data Preparation        — clean, encode, engineer features
#   4. Modeling                — train regression/classification models
#   5. Evaluation              — measure performance with proper metrics
#   6. Deployment              — serve predictions via an API
#
# This script covers steps 1 and 2.
# -----------------------------------------------------------------------------

import os
import requests
import pandas as pd

# -----------------------------------------------------------------------------
# STEP 1 — BUSINESS UNDERSTANDING
# WFP monitors food prices in vulnerable markets worldwide.
# Predicting commodity prices enables early warning of food crises.
# Our ML goal: predict the USD price of a commodity in a given market
#              based on location, commodity type, and time features.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# STEP 2 — DATA UNDERSTANDING
# -----------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATA_FILE = os.path.join(DATA_DIR, "wfpvam_foodprices.csv")
DATA_URL = (
    "https://data.humdata.org/dataset/4fdcd4dc-5c2f-43af-a1e4-93c9b6539a27"
    "/resource/12d7c8e3-eff9-4db0-93b7-726825c4fe9a/download/wfpvam_foodprices.csv"
)


def download_data():
    """Download the WFP food prices CSV if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DATA_FILE):
        print(f"[INFO] Data already present at: {DATA_FILE}")
        return
    print("[INFO] Downloading WFP Global Food Prices dataset ...")
    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()
    with open(DATA_FILE, "wb") as f:
        f.write(response.content)
    print(f"[INFO] Saved to: {DATA_FILE}")


def load_data():
    """Load the CSV into a pandas DataFrame."""
    df = pd.read_csv(DATA_FILE, low_memory=False)
    return df


def explore_data(df: pd.DataFrame):
    """Print a structured overview of the dataset."""

    print("\n" + "=" * 60)
    print("WFP GLOBAL FOOD PRICES — DATA UNDERSTANDING")
    print("=" * 60)

    # Shape
    print(f"\n[Shape] {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Columns and dtypes
    print("\n[Columns & Types]")
    print(df.dtypes.to_string())

    # Sample rows
    print("\n[Sample — first 3 rows]")
    print(df.head(3).to_string(index=False))

    # Missing values
    print("\n[Missing Values]")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  No missing values found.")
    else:
        pct = (missing / len(df) * 100).round(2)
        print(pd.DataFrame({"missing": missing, "pct (%)": pct}).to_string())

    # Key categorical columns
    for col in ["adm0_name", "cm_name", "pt_name", "cur_name", "um_name"]:
        if col in df.columns:
            n = df[col].nunique()
            sample = df[col].dropna().unique()[:5].tolist()
            print(f"\n[{col}] {n} unique values — e.g. {sample}")

    # Target variable
    price_col = "mp_price"
    if price_col in df.columns:
        print(f"\n[Target: {price_col}]")
        print(df[price_col].describe().round(4).to_string())

    # Date range
    if "mp_year" in df.columns and "mp_month" in df.columns:
        print(f"\n[Time range] {int(df['mp_year'].min())} — {int(df['mp_year'].max())}")

    print("\n" + "=" * 60)
    print("ML PROBLEM FRAMING")
    print("=" * 60)
    print("  Type    : Supervised learning — Regression")
    print("  Target  : mp_price (commodity price in local currency)")
    print("  Features: country, market, commodity, currency, unit, year, month")
    print("  Goal    : predict food commodity prices to support early warning")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    download_data()
    df = load_data()
    explore_data(df)
