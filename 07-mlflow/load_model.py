# =============================================================================
# MODULE 07 — Load Model from MLflow Registry
# Demonstrates loading the production model by alias and scoring new data.
# Run AFTER registry.py:
#   python 07-mlflow/load_model.py
# =============================================================================

import os
import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc

# -----------------------------------------------------------------------------
# CONNECT
# -----------------------------------------------------------------------------

DB_PATH = os.path.join(os.path.dirname(__file__), "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath(DB_PATH)}")

REGISTERED_NAME = "maize-price-alert-xgboost"
YEAR_MIN        = 1990

# The MLflow-logged XGBoost model is the raw booster — no preprocessor.
# We load the preprocessor from the pickle artifact saved in Module 06.
ARTIFACT_FILE = os.path.join(
    os.path.dirname(__file__), "..", "05-deployment", "model", "xgb_pipeline.pkl"
)

# -----------------------------------------------------------------------------
# LOAD PRODUCTION MODEL BY ALIAS
# -----------------------------------------------------------------------------

print(f"[INFO] Loading model '{REGISTERED_NAME}@production' from registry ...")
model_uri = f"models:/{REGISTERED_NAME}@production"
model     = mlflow.pyfunc.load_model(model_uri)
print(f"[INFO] Model loaded. Flavor: {list(model.metadata.flavors.keys())}")

# Load the preprocessor saved alongside the model in Module 06
print(f"[INFO] Loading preprocessor from: {ARTIFACT_FILE}")
with open(ARTIFACT_FILE, "rb") as f:
    artifact = pickle.load(f)
preprocessor = artifact["preprocessor"]
print("[INFO] Preprocessor loaded.")

# -----------------------------------------------------------------------------
# PREPARE TEST OBSERVATIONS
# -----------------------------------------------------------------------------

CAT_FEATURES = ["adm0_name", "cur_name", "adm1_name"]
NUM_FEATURES = ["year_norm", "month_sin", "month_cos"]

def build_features(observations: list) -> pd.DataFrame:
    rows = []
    for obs in observations:
        year_norm = obs["mp_year"] - YEAR_MIN
        month     = obs["mp_month"]
        rows.append({
            "adm0_name" : obs["adm0_name"],
            "cur_name"  : obs["cur_name"],
            "adm1_name" : obs.get("adm1_name", "Unknown"),
            "year_norm" : float(year_norm),
            "month_sin" : float(np.sin(2 * np.pi * month / 12)),
            "month_cos" : float(np.cos(2 * np.pi * month / 12)),
        })
    return pd.DataFrame(rows)


test_observations = [
    {"adm0_name": "Rwanda",   "cur_name": "RWF", "adm1_name": "Kigali City", "mp_year": 2020, "mp_month": 6},
    {"adm0_name": "Mali",     "cur_name": "XOF", "adm1_name": "Sikasso",     "mp_year": 2019, "mp_month": 3},
    {"adm0_name": "Zambia",   "cur_name": "ZMW", "adm1_name": "Unknown",     "mp_year": 2021, "mp_month": 11},
    {"adm0_name": "Ethiopia", "cur_name": "ETB", "adm1_name": "Unknown",     "mp_year": 2020, "mp_month": 1},
]

X_raw   = build_features(test_observations)
X_test  = pd.DataFrame(
    preprocessor.transform(X_raw),
    columns=CAT_FEATURES + NUM_FEATURES
)

# -----------------------------------------------------------------------------
# PREDICT
# -----------------------------------------------------------------------------

print("\n[INFO] Running predictions ...")
predictions = model.predict(X_test)

print("\n" + "=" * 55)
print("PREDICTIONS FROM REGISTRY MODEL")
print("=" * 55)
print(f"  {'Country':<12} {'Year':>6} {'Month':>6} {'Prediction':>12}")
print("  " + "-" * 40)
for obs, pred in zip(test_observations, predictions):
    label = "HIGH PRICE" if int(pred) == 1 else "Normal"
    print(f"  {obs['adm0_name']:<12} {obs['mp_year']:>6} {obs['mp_month']:>6} {label:>12}")

print("\n[INFO] load_model.py complete.")