# =============================================================================
# MODULE 07 — MLflow Experiment Tracking
# Logs all models trained in Modules 03–06 into a single MLflow experiment.
# Run mlflow ui after this script to explore results in the browser.
#
# Usage:
#   python 07-mlflow/tracking.py
#   mlflow ui --backend-store-uri 07-mlflow/mlruns
# Then open: http://localhost:5000
# =============================================================================

import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import xgboost as xgb

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------

DATA_FILE   = os.path.join(os.path.dirname(__file__), "..", "data", "wfp_maize_clean.csv")
DB_PATH = os.path.join(os.path.dirname(__file__), "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath(DB_PATH)}")
mlflow.set_experiment("wfp-maize-price-alert")

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

train = df[df["mp_year"] < 2019].copy()
val   = df[df["mp_year"] >= 2019].copy()

X_train_raw = train[ALL_FEATURES]
y_train     = train[TARGET]
X_val_raw   = val[ALL_FEATURES]
y_val       = val[TARGET]

# Preprocessors
ohe_pre = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ("num", "passthrough", NUM_FEATURES),
])
ord_pre = ColumnTransformer(transformers=[
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_FEATURES),
    ("num", "passthrough", NUM_FEATURES),
])
ord_pre.fit(X_train_raw)
X_train_ord = ord_pre.transform(X_train_raw)
X_val_ord   = ord_pre.transform(X_val_raw)

print(f"[INFO] Train: {len(train):,} | Val: {len(val):,}")

# -----------------------------------------------------------------------------
# HELPER — log a run
# -----------------------------------------------------------------------------

def log_run(run_name, params, model, X_val, y_val, use_proba=True, log_model=False, model_obj=None):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        if use_proba:
            y_proba = model.predict_proba(X_val)[:, 1]
        else:
            y_proba = model.predict(X_val).astype(float)

        y_pred = model.predict(X_val)
        auc    = roc_auc_score(y_val, y_proba)
        f1     = f1_score(y_val, y_pred)
        acc    = accuracy_score(y_val, y_pred)

        mlflow.log_metric("val_roc_auc", round(auc, 4))
        mlflow.log_metric("val_f1",      round(f1, 4))
        mlflow.log_metric("val_accuracy",round(acc, 4))

        if log_model and model_obj is not None:
            mlflow.sklearn.log_model(model_obj, artifact_path="model")

        print(f"  [{run_name}] AUC={auc:.4f}  F1={f1:.4f}  ACC={acc:.4f}")
        return auc, f1

# -----------------------------------------------------------------------------
# RUN 1 — Logistic Regression (best from Module 03)
# -----------------------------------------------------------------------------

print("\n[INFO] Logging Logistic Regression ...")
lr_pipe = Pipeline([
    ("pre",   ohe_pre),
    ("model", LogisticRegression(C=0.1, solver="saga", max_iter=1000, random_state=42)),
])
lr_pipe.fit(X_train_raw, y_train)
log_run(
    run_name="logistic_regression_C0.1",
    params={"model": "LogisticRegression", "C": 0.1, "solver": "saga",
            "encoding": "OneHot", "split_year": 2019},
    model=lr_pipe,
    X_val=X_val_raw,
    y_val=y_val,
    log_model=True,
    model_obj=lr_pipe,
)

# -----------------------------------------------------------------------------
# RUN 2 — Decision Tree (best depth from Module 06)
# -----------------------------------------------------------------------------

print("[INFO] Logging Decision Tree ...")
dt_pipe = Pipeline([
    ("pre",   ord_pre),
    ("model", DecisionTreeClassifier(max_depth=15, random_state=42)),
])
dt_pipe.fit(X_train_raw, y_train)
log_run(
    run_name="decision_tree_depth15",
    params={"model": "DecisionTree", "max_depth": 15,
            "encoding": "Ordinal", "split_year": 2019},
    model=dt_pipe,
    X_val=X_val_raw,
    y_val=y_val,
)

# -----------------------------------------------------------------------------
# RUN 3 — Random Forest (best from Module 06)
# -----------------------------------------------------------------------------

print("[INFO] Logging Random Forest ...")
rf_pipe = Pipeline([
    ("pre",   ord_pre),
    ("model", RandomForestClassifier(
        n_estimators=100, max_depth=10, n_jobs=-1, random_state=42
    )),
])
rf_pipe.fit(X_train_raw, y_train)
log_run(
    run_name="random_forest_n100_d10",
    params={"model": "RandomForest", "n_estimators": 100, "max_depth": 10,
            "encoding": "Ordinal", "split_year": 2019},
    model=rf_pipe,
    X_val=X_val_raw,
    y_val=y_val,
)

# -----------------------------------------------------------------------------
# RUN 4 — XGBoost (best from Module 06 early stopping)
# -----------------------------------------------------------------------------

print("[INFO] Logging XGBoost ...")
xgb_model = xgb.XGBClassifier(
    max_depth=8, learning_rate=0.03, n_estimators=95,
    subsample=0.8, colsample_bytree=0.7,
    eval_metric="logloss", random_state=42, n_jobs=-1,
)
xgb_model.fit(X_train_ord, y_train)

with mlflow.start_run(run_name="xgboost_d8_lr0.03_n95"):
    params = {
        "model": "XGBoost", "max_depth": 8, "learning_rate": 0.03,
        "n_estimators": 95, "subsample": 0.8, "colsample_bytree": 0.7,
        "encoding": "Ordinal", "split_year": 2019,
    }
    mlflow.log_params(params)

    y_proba = xgb_model.predict_proba(X_val_ord)[:, 1]
    y_pred  = xgb_model.predict(X_val_ord)
    auc     = roc_auc_score(y_val, y_proba)
    f1      = f1_score(y_val, y_pred)
    acc     = accuracy_score(y_val, y_pred)

    mlflow.log_metric("val_roc_auc",  round(auc, 4))
    mlflow.log_metric("val_f1",       round(f1, 4))
    mlflow.log_metric("val_accuracy", round(acc, 4))
    mlflow.xgboost.log_model(xgb_model, artifact_path="model")

    print(f"  [xgboost_d8_lr0.03_n95] AUC={auc:.4f}  F1={f1:.4f}  ACC={acc:.4f}")

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------

print("\n" + "=" * 55)
print("ALL RUNS LOGGED TO MLFLOW")
print("=" * 55)
print(f"  Tracking URI : {mlflow.get_tracking_uri()}")
print(f"  Experiment   : wfp-maize-price-alert")
print()
print("  To explore results in the browser:")
print(f"  mlflow ui --backend-store-uri sqlite:///{os.path.abspath(DB_PATH)}")
print("  Then open: http://localhost:5000")
print()
print("[INFO] tracking.py complete.")