# =============================================================================
# MODULE 06 — XGBoost with Early Stopping & Hyperparameter Tuning
# Goal   : Find the best XGBoost model, save it for deployment
# Strategy: Use a secondary validation split within training data for
#           early stopping, then evaluate on the held-out temporal val set
# =============================================================================

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------

DATA_FILE  = os.path.join(os.path.dirname(__file__), "..", "data", "wfp_maize_clean.csv")
PLOTS_DIR  = os.path.join(os.path.dirname(__file__), "plots")
MODEL_FILE = os.path.join(os.path.dirname(__file__), "..", "05-deployment", "model", "xgb_pipeline.pkl")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)

# -----------------------------------------------------------------------------
# LOAD & PREPARE
# -----------------------------------------------------------------------------

print("[INFO] Loading data ...")
df = pd.read_csv(DATA_FILE, parse_dates=["date"])

country_median   = df.groupby("adm0_name")["log_price"].median().rename("country_median")
df               = df.join(country_median, on="adm0_name")
df["high_price"] = (df["log_price"] > df["country_median"]).astype(int)
df["year_norm"]  = df["mp_year"] - df["mp_year"].min()
df["month_sin"]  = np.sin(2 * np.pi * df["mp_month"] / 12)
df["month_cos"]  = np.cos(2 * np.pi * df["mp_month"] / 12)

CAT_FEATURES = ["adm0_name", "cur_name", "adm1_name"]
NUM_FEATURES = ["year_norm", "month_sin", "month_cos"]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES
TARGET       = "high_price"
YEAR_MIN     = int(df["mp_year"].min())

# Temporal splits:
#   train    : < 2017  (used to train XGBoost)
#   es_val   : 2017–2018 (used for early stopping within XGBoost)
#   test_val : >= 2019  (held-out temporal validation — never seen during tuning)

train_mask  = df["mp_year"] < 2017
es_mask     = (df["mp_year"] >= 2017) & (df["mp_year"] < 2019)
val_mask    = df["mp_year"] >= 2019

train  = df[train_mask].copy()
es_val = df[es_mask].copy()
val    = df[val_mask].copy()

print(f"[INFO] Train: {len(train):,} | Early-stop val: {len(es_val):,} | Test val: {len(val):,}")

# -----------------------------------------------------------------------------
# PREPROCESSING
# -----------------------------------------------------------------------------

preprocessor = ColumnTransformer(transformers=[
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_FEATURES),
    ("num", "passthrough", NUM_FEATURES),
])

preprocessor.fit(train[ALL_FEATURES])

X_train = preprocessor.transform(train[ALL_FEATURES])
X_es    = preprocessor.transform(es_val[ALL_FEATURES])
X_val   = preprocessor.transform(val[ALL_FEATURES])

y_train = train[TARGET].values
y_es    = es_val[TARGET].values
y_val   = val[TARGET].values

# -----------------------------------------------------------------------------
# XGBOOST WITH EARLY STOPPING
# early_stopping_rounds=20: stop if val AUC doesn't improve for 20 rounds
# This avoids overfitting to the training period without manually sweeping n_estimators
# -----------------------------------------------------------------------------

print("\n[INFO] Training XGBoost with early stopping ...")

configs = [
    {"max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.7},
    {"max_depth": 6, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.8},
    {"max_depth": 8, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.7},
]

print(f"\n{'Config':<45} {'Best iter':>10} {'ES-AUC':>8} {'Val-AUC':>9}")
print("-" * 75)

best_val_auc   = -1
best_model     = None
best_cfg       = None
best_n_iters   = None

for cfg in configs:
    model = xgb.XGBClassifier(
        **cfg,
        n_estimators=1000,
        eval_metric="auc",
        early_stopping_rounds=20,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_es, y_es)],
        verbose=False,
    )

    n_iters  = model.best_iteration + 1
    es_auc   = roc_auc_score(y_es,  model.predict_proba(X_es)[:, 1])
    val_auc  = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    cfg_str  = f"d={cfg['max_depth']} lr={cfg['learning_rate']} ss={cfg['subsample']}"

    marker = " ◄" if val_auc > best_val_auc else ""
    print(f"{cfg_str:<45} {n_iters:>10} {es_auc:>8.4f} {val_auc:>9.4f}{marker}")

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_model   = model
        best_cfg     = cfg
        best_n_iters = n_iters

# -----------------------------------------------------------------------------
# FINAL EVALUATION
# -----------------------------------------------------------------------------

y_proba_val = best_model.predict_proba(X_val)[:, 1]
y_pred_val  = best_model.predict(X_val)

f1  = f1_score(y_val, y_pred_val)
auc = roc_auc_score(y_val, y_proba_val)

print(f"\n[Best config] {best_cfg}")
print(f"[Best n_iters] {best_n_iters}")
print(f"\n[Final Temporal Val Metrics]")
print(f"  ROC-AUC : {auc:.4f}")
print(f"  F1      : {f1:.4f}")
print()
print(classification_report(y_val, y_pred_val,
                             target_names=["Normal (0)", "High price (1)"]))

# -----------------------------------------------------------------------------
# LEARNING CURVE — AUC vs boosting round on early-stop val set
# -----------------------------------------------------------------------------

evals_result = best_model.evals_result()
es_auc_curve = evals_result["validation_0"]["auc"]

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(range(1, len(es_auc_curve) + 1), es_auc_curve,
        color="#2196F3", linewidth=1.5, label="Early-stop val AUC")
ax.axvline(best_n_iters, color="red", linestyle="--", linewidth=1.2,
           label=f"Best iteration = {best_n_iters}")
ax.set_xlabel("Boosting Round", fontsize=12)
ax.set_ylabel("AUC", fontsize=12)
ax.set_title("XGBoost Learning Curve (Early Stopping on 2017–2018 data)", fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "21_xgb_learning_curve.png"), dpi=150)
plt.close()
print("[INFO] Saved: 21_xgb_learning_curve.png")

# Feature importance
importance = pd.Series(
    best_model.feature_importances_,
    index=ALL_FEATURES
).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 4))
importance.sort_values().plot(kind="barh", ax=ax, color="#FF9800")
ax.set_xlabel("Feature Importance (gain)", fontsize=12)
ax.set_title("XGBoost (Early Stopping) — Feature Importance", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "22_xgb_es_feature_importance.png"), dpi=150)
plt.close()
print("[INFO] Saved: 22_xgb_es_feature_importance.png")

# -----------------------------------------------------------------------------
# SAVE BEST XGB MODEL + PREPROCESSOR
# Overwrites the logistic model in 05-deployment/model/ — this is now our
# best model and will be served by the FastAPI app
# -----------------------------------------------------------------------------

artifact = {
    "model"       : best_model,
    "preprocessor": preprocessor,
    "year_min"    : YEAR_MIN,
    "cat_features": CAT_FEATURES,
    "num_features": NUM_FEATURES,
    "model_type"  : "xgboost",
    "val_auc"     : round(auc, 4),
}

with open(MODEL_FILE, "wb") as f:
    pickle.dump(artifact, f)

print(f"\n[INFO] Best XGBoost model saved to: {MODEL_FILE}")
print(f"[INFO] Val AUC: {auc:.4f}")
print("[INFO] train_xgboost.py complete.")