# =============================================================================
# MODULE 03 — Classification: Logistic Regression
# Task    : Predict whether a maize market is in a high-price state
# Target  : high_price = 1 if log_price > country median, else 0
# Features: country, currency, region, year, month (cyclic)
# =============================================================================

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# -----------------------------------------------------------------------------
# LOAD
# -----------------------------------------------------------------------------

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "wfp_maize_clean.csv")
print("[INFO] Loading clean maize dataset ...")
df = pd.read_csv(DATA_FILE, parse_dates=["date"])
print(f"[INFO] Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# -----------------------------------------------------------------------------
# BUILD BINARY TARGET
# For each country, compute the median log_price over the full dataset.
# A record is labeled 1 (high price) if its log_price exceeds that country median.
# This normalizes for currency scale — the label reflects relative price stress.
# -----------------------------------------------------------------------------

country_median = df.groupby("adm0_name")["log_price"].median().rename("country_median")
df = df.join(country_median, on="adm0_name")
df["high_price"] = (df["log_price"] > df["country_median"]).astype(int)

print(f"\n[Target distribution]")
vc = df["high_price"].value_counts()
print(f"  high_price=0 (normal)  : {vc[0]:,} ({vc[0]/len(df)*100:.1f}%)")
print(f"  high_price=1 (elevated): {vc[1]:,} ({vc[1]/len(df)*100:.1f}%)")

# -----------------------------------------------------------------------------
# FEATURE ENGINEERING (same as Module 02)
# -----------------------------------------------------------------------------

df["year_norm"]  = df["mp_year"] - df["mp_year"].min()
df["month_sin"]  = np.sin(2 * np.pi * df["mp_month"] / 12)
df["month_cos"]  = np.cos(2 * np.pi * df["mp_month"] / 12)

CAT_FEATURES = ["adm0_name", "cur_name", "adm1_name"]
NUM_FEATURES = ["year_norm", "month_sin", "month_cos"]
TARGET       = "high_price"

# -----------------------------------------------------------------------------
# TRAIN / VALIDATION SPLIT (same temporal split as Module 02)
# -----------------------------------------------------------------------------

SPLIT_YEAR = 2019
train = df[df["mp_year"] < SPLIT_YEAR].copy()
val   = df[df["mp_year"] >= SPLIT_YEAR].copy()

print(f"\n[Split] Train: {len(train):,} rows | Val: {len(val):,} rows")
print(f"  Train class balance: {train[TARGET].mean():.3f} positive rate")
print(f"  Val   class balance: {val[TARGET].mean():.3f} positive rate")

X_train = train[CAT_FEATURES + NUM_FEATURES]
y_train = train[TARGET]
X_val   = val[CAT_FEATURES + NUM_FEATURES]
y_val   = val[TARGET]

# -----------------------------------------------------------------------------
# PREPROCESSING PIPELINE
# -----------------------------------------------------------------------------

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ("num", "passthrough", NUM_FEATURES),
])

# -----------------------------------------------------------------------------
# MODEL — LOGISTIC REGRESSION
# C is the inverse of regularization strength (higher C = less regularization)
# solver="saga" handles large sparse OHE matrices efficiently
# max_iter=1000 ensures convergence with many categories
# -----------------------------------------------------------------------------

print("\n" + "=" * 55)
print("LOGISTIC REGRESSION — C sweep")
print("=" * 55)
print(f"  {'C':>8} | {'Accuracy':>10} | {'Precision':>10} | {'Recall':>8} | {'F1':>8}")
print("  " + "-" * 52)

best_C      = None
best_f1     = -1
best_model  = None

for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            C=C, solver="saga", max_iter=1000, random_state=42
        )),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_val)

    acc  = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec  = recall_score(y_val, y_pred, zero_division=0)
    f1   = f1_score(y_val, y_pred, zero_division=0)

    marker = " ◄ best" if f1 > best_f1 else ""
    print(f"  {C:>8.2f} | {acc:>10.4f} | {prec:>10.4f} | {rec:>8.4f} | {f1:>8.4f}{marker}")

    if f1 > best_f1:
        best_f1    = f1
        best_C     = C
        best_model = pipe

# -----------------------------------------------------------------------------
# DETAILED REPORT FOR BEST MODEL
# -----------------------------------------------------------------------------

y_pred_best  = best_model.predict(X_val)
y_proba_best = best_model.predict_proba(X_val)[:, 1]

print(f"\n[Best C = {best_C}]")
print("\n[Classification Report]")
print(classification_report(y_val, y_pred_best,
                             target_names=["Normal (0)", "High price (1)"]))

cm = confusion_matrix(y_val, y_pred_best)
print("[Confusion Matrix]")
print(f"  {'':20} Predicted 0   Predicted 1")
print(f"  {'Actual 0':20} {cm[0,0]:>11,}   {cm[0,1]:>11,}")
print(f"  {'Actual 1':20} {cm[1,0]:>11,}   {cm[1,1]:>11,}")

# -----------------------------------------------------------------------------
# FEATURE IMPORTANCE — top logistic regression coefficients
# Positive coef → pushes toward high_price=1
# Negative coef → pushes toward high_price=0
# -----------------------------------------------------------------------------

feature_names = (
    best_model.named_steps["preprocessor"]
    .named_transformers_["cat"]
    .get_feature_names_out(CAT_FEATURES)
    .tolist()
    + NUM_FEATURES
)
coefs = best_model.named_steps["model"].coef_[0]
importance = pd.Series(coefs, index=feature_names).sort_values()

print(f"\n[Top 10 features pushing toward HIGH price (coef > 0)]")
print(importance.tail(10).round(4).to_string())

print(f"\n[Top 10 features pushing toward NORMAL price (coef < 0)]")
print(importance.head(10).round(4).to_string())

# -----------------------------------------------------------------------------
# SAVE PREDICTIONS FOR predict.py
# -----------------------------------------------------------------------------

val_results = val[["adm0_name", "mp_year", "mp_month", "log_price", "high_price"]].copy()
val_results["high_price_pred"]  = y_pred_best
val_results["high_price_proba"] = y_proba_best.round(4)

out_path = os.path.join(os.path.dirname(__file__), "..", "data", "val_clf_predictions.csv")
val_results.to_csv(out_path, index=False)
print(f"\n[INFO] Saved validation predictions to: {out_path}")
print("[INFO] Training complete.")