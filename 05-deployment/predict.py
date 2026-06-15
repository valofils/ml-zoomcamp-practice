# =============================================================================
# MODULE 05 — Standalone Prediction Function
# Loads the serialized pipeline and scores a single observation.
# This module is imported by app.py — it has no FastAPI dependency.
# =============================================================================

import os
import pickle
import numpy as np
import pandas as pd

MODEL_FILE = os.path.join(os.path.dirname(__file__), "model", "logistic_pipeline.pkl")


def load_model():
    """Load the serialized pipeline artifact from disk."""
    with open(MODEL_FILE, "rb") as f:
        artifact = pickle.load(f)
    return artifact


def build_features(observation: dict, year_min: int) -> pd.DataFrame:
    """
    Convert a raw observation dict into a feature DataFrame
    that matches the format expected by the pipeline.

    Expected keys in observation:
        adm0_name  (str) : country name
        cur_name   (str) : currency code
        adm1_name  (str) : sub-national region (use "Unknown" if not available)
        mp_year    (int) : year of the price observation
        mp_month   (int) : month (1–12)
    """
    year_norm = observation["mp_year"] - year_min
    month     = observation["mp_month"]

    row = {
        "adm0_name" : observation["adm0_name"],
        "cur_name"  : observation["cur_name"],
        "adm1_name" : observation.get("adm1_name", "Unknown"),
        "year_norm" : year_norm,
        "month_sin" : np.sin(2 * np.pi * month / 12),
        "month_cos" : np.cos(2 * np.pi * month / 12),
    }
    return pd.DataFrame([row])


def predict(observation: dict) -> dict:
    """
    Score a single observation and return prediction + probability.

    Returns:
        {
            "high_price"      : 0 or 1,
            "high_price_proba": float (probability of high price),
            "alert"           : str  (human-readable label)
        }
    """
    artifact = load_model()
    pipeline = artifact["pipeline"]
    year_min = artifact["year_min"]

    features = build_features(observation, year_min)
    pred     = int(pipeline.predict(features)[0])
    proba    = float(pipeline.predict_proba(features)[0][1])

    return {
        "high_price"       : pred,
        "high_price_proba" : round(proba, 4),
        "alert"            : "HIGH PRICE ALERT" if pred == 1 else "Normal price level",
    }


# -----------------------------------------------------------------------------
# QUICK SMOKE TEST — run this file directly to verify the model loads correctly
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        {
            "adm0_name": "Rwanda",
            "cur_name" : "RWF",
            "adm1_name": "Kigali City",
            "mp_year"  : 2020,
            "mp_month" : 6,
        },
        {
            "adm0_name": "Mali",
            "cur_name" : "XOF",
            "adm1_name": "Sikasso",
            "mp_year"  : 2019,
            "mp_month" : 3,
        },
        {
            "adm0_name": "Zambia",
            "cur_name" : "ZMW",
            "adm1_name": "Unknown",
            "mp_year"  : 2021,
            "mp_month" : 11,
        },
    ]

    print("[INFO] Loading model ...")
    artifact = load_model()
    print(f"[INFO] Model loaded. year_min = {artifact['year_min']}")
    print()

    for obs in test_cases:
        result = predict(obs)
        print(f"Input : {obs['adm0_name']} | {obs['mp_year']}-{obs['mp_month']:02d}")
        print(f"Output: {result}")
        print()
