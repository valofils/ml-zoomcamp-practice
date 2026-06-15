# =============================================================================
# MODULE 05 — Model Serialization
# Trains the best logistic regression from Module 03 on the full training set
# and saves the pipeline to disk using pickle.
# The saved model is what the FastAPI service will load and serve.
# =============================================================================

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, f1_score

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------

DATA_FILE  = os.path.join(os.path.dirname(__file__), "..", "data", "wfp_maize_clean.csv")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "model")
MODEL_FILE = os.path.join(MODEL_DIR, "logistic_pipeline.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# LOAD & PREPARE
# -----------------------------------------------------------------------------

print("[INFO] Loading data ...")
df = pd.read_csv(DATA_FILE, parse_dates=["date"])
print(f"[INFO] Shape: {df.shape[0]:,} rows")

# Target
country_median = df.groupby("adm0_name")["log_price"].median().rename("country_median")
df = df.join(country_median, on="adm0_name")
df["high_price"] = (df["log_price"] > df["country_median"]).astype(int)

# Features
df["year_norm"] = df["mp_year"] - df["mp_year"].min()
df["month_sin"] = np.sin(2 * np.pi * df["mp_month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["mp_month"] / 12)

CAT_FEATURES = ["adm0_name", "cur_name", "adm1_name"]
NUM_FEATURES = ["year_norm", "month_sin", "month_cos"]
TARGET       = "high_price"

# Time-based split
SPLIT_YEAR = 2019
train = df[df["mp_year"] < SPLIT_YEAR].copy()
val   = df[df["mp_year"] >= SPLIT_YEAR].copy()

X_train = train[CAT_FEATURES + NUM_FEATURES]
y_train = train[TARGET]
X_val   = val[CAT_FEATURES + NUM_FEATURES]
y_val   = val[TARGET]

print(f"[INFO] Train: {len(train):,} | Val: {len(val):,}")

# -----------------------------------------------------------------------------
# BUILD AND TRAIN PIPELINE
# Best hyperparameters from Module 03: C=0.1, solver=saga
# -----------------------------------------------------------------------------

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ("num", "passthrough", NUM_FEATURES),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(C=0.1, solver="saga", max_iter=1000, random_state=42)),
])

print("[INFO] Training pipeline ...")
pipeline.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# EVALUATE ON VALIDATION SET
# -----------------------------------------------------------------------------

y_proba = pipeline.predict_proba(X_val)[:, 1]
y_pred  = pipeline.predict(X_val)

auc = roc_auc_score(y_val, y_proba)
f1  = f1_score(y_val, y_pred)

print(f"\n[Validation metrics]")
print(f"  ROC-AUC : {auc:.4f}")
print(f"  F1      : {f1:.4f}")

# Store the year offset used in feature engineering so the API can reproduce it
year_min = int(df["mp_year"].min())

# -----------------------------------------------------------------------------
# SERIALIZE
# We save both the pipeline and the metadata needed to reproduce features
# -----------------------------------------------------------------------------

artifact = {
    "pipeline" : pipeline,
    "year_min" : year_min,
    "cat_features": CAT_FEATURES,
    "num_features": NUM_FEATURES,
}

with open(MODEL_FILE, "wb") as f:
    pickle.dump(artifact, f)

print(f"\n[INFO] Model saved to: {MODEL_FILE}")
print(f"[INFO] year_min stored: {year_min}")
print("[INFO] Serialization complete.")
