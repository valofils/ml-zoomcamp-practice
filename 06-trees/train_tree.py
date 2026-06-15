# =============================================================================
# MODULE 06 — Decision Trees & Ensemble Learning
# Models : Decision Tree, Random Forest, XGBoost
# Task   : Same binary classification as Module 03 (high_price)
# Goal   : Show that tree-based models close the AUC gap left by logistic reg
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import xgboost as xgb

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "wfp_maize_clean.csv")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# LOAD & PREPARE
# -----------------------------------------------------------------------------

print("[INFO] Loading data ...")
df = pd.read_csv(DATA_FILE, parse_dates=["date"])

country_median  = df.groupby("adm0_name")["log_price"].median().rename("country_median")
df              = df.join(country_median, on="adm0_name")
df["high_price"] = (df["log_price"] > df["country_median"]).astype(int)
df["year_norm"]  = df["mp_year"] - df["mp_year"].min()
df["month_sin"]  = np.sin(2 * np.pi * df["mp_month"] / 12)
df["month_cos"]  = np.cos(2 * np.pi * df["mp_month"] / 12)

CAT_FEATURES = ["adm0_name", "cur_name", "adm1_name"]
NUM_FEATURES = ["year_norm", "month_sin", "month_cos"]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES
TARGET       = "high_price"

SPLIT_YEAR = 2019
train = df[df["mp_year"] < SPLIT_YEAR].copy()
val   = df[df["mp_year"] >= SPLIT_YEAR].copy()

X_train = train[ALL_FEATURES]
y_train = train[TARGET]
X_val   = val[ALL_FEATURES]
y_val   = val[TARGET]

print(f"[INFO] Train: {len(train):,} | Val: {len(val):,}")

# -----------------------------------------------------------------------------
# PREPROCESSING
# Tree models need OrdinalEncoder (not OHE) — trees split on integer codes,
# OHE would explode dimensionality unnecessarily for ensembles.
# -----------------------------------------------------------------------------

preprocessor = ColumnTransformer(transformers=[
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_FEATURES),
    ("num", "passthrough", NUM_FEATURES),
])

# -----------------------------------------------------------------------------
# HELPER
# -----------------------------------------------------------------------------

def evaluate(name, y_true, y_pred, y_proba):
    auc = roc_auc_score(y_true, y_proba)
    f1  = f1_score(y_true, y_pred)
    print(f"  {name:<30} AUC={auc:.4f}  F1={f1:.4f}")
    return auc, f1

# -----------------------------------------------------------------------------
# MODEL 1 — DECISION TREE (depth sweep)
# A single decision tree is fast and interpretable.
# max_depth controls bias-variance trade-off.
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("MODEL 1 — Decision Tree (max_depth sweep)")
print("=" * 60)

dt_results = []
for depth in [3, 5, 10, 15, 20, None]:
    pipe = Pipeline([
        ("pre", preprocessor),
        ("model", DecisionTreeClassifier(max_depth=depth, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    y_pred  = pipe.predict(X_val)
    y_proba = pipe.predict_proba(X_val)[:, 1]
    auc, f1 = evaluate(f"depth={depth}", y_val, y_pred, y_proba)
    dt_results.append({"depth": str(depth), "auc": auc, "f1": f1})

best_dt = max(dt_results, key=lambda x: x["auc"])
print(f"\n  Best depth by AUC: {best_dt['depth']}  (AUC={best_dt['auc']:.4f})")

# -----------------------------------------------------------------------------
# MODEL 2 — RANDOM FOREST (n_estimators sweep)
# Ensemble of trees — reduces variance through bagging.
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("MODEL 2 — Random Forest (n_estimators sweep, max_depth=10)")
print("=" * 60)

rf_results = []
for n in [50, 100, 200]:
    pipe = Pipeline([
        ("pre", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=n, max_depth=10,
            n_jobs=-1, random_state=42
        )),
    ])
    pipe.fit(X_train, y_train)
    y_pred  = pipe.predict(X_val)
    y_proba = pipe.predict_proba(X_val)[:, 1]
    auc, f1 = evaluate(f"n_estimators={n}", y_val, y_pred, y_proba)
    rf_results.append({"n": n, "auc": auc, "f1": f1})

best_rf_row = max(rf_results, key=lambda x: x["auc"])
print(f"\n  Best n_estimators by AUC: {best_rf_row['n']}  (AUC={best_rf_row['auc']:.4f})")

# -----------------------------------------------------------------------------
# MODEL 3 — XGBOOST (key hyperparameter sweep)
# Gradient boosting — builds trees sequentially, each correcting prior errors.
# XGBoost accepts OrdinalEncoded features natively (enable_categorical=False).
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("MODEL 3 — XGBoost (hyperparameter sweep)")
print("=" * 60)

# Pre-transform features once for XGBoost (faster than pipeline for sweep)
preprocessor.fit(X_train)
X_train_t = preprocessor.transform(X_train)
X_val_t   = preprocessor.transform(X_val)

xgb_configs = [
    {"max_depth": 3,  "n_estimators": 100, "learning_rate": 0.1},
    {"max_depth": 5,  "n_estimators": 100, "learning_rate": 0.1},
    {"max_depth": 5,  "n_estimators": 200, "learning_rate": 0.05},
    {"max_depth": 6,  "n_estimators": 200, "learning_rate": 0.05},
    {"max_depth": 6,  "n_estimators": 300, "learning_rate": 0.03},
]

xgb_results = []
for cfg in xgb_configs:
    model = xgb.XGBClassifier(
        **cfg,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_t, y_train)
    y_pred  = model.predict(X_val_t)
    y_proba = model.predict_proba(X_val_t)[:, 1]
    auc, f1 = evaluate(
        f"d={cfg['max_depth']} n={cfg['n_estimators']} lr={cfg['learning_rate']}",
        y_val, y_pred, y_proba
    )
    xgb_results.append({**cfg, "auc": auc, "f1": f1, "model": model})

best_xgb = max(xgb_results, key=lambda x: x["auc"])
print(f"\n  Best XGBoost config by AUC:")
print(f"    max_depth={best_xgb['max_depth']}  n_estimators={best_xgb['n_estimators']}  lr={best_xgb['learning_rate']}")
print(f"    AUC={best_xgb['auc']:.4f}  F1={best_xgb['f1']:.4f}")

# -----------------------------------------------------------------------------
# FULL REPORT — best XGBoost
# -----------------------------------------------------------------------------

y_pred_xgb  = best_xgb["model"].predict(X_val_t)
y_proba_xgb = best_xgb["model"].predict_proba(X_val_t)[:, 1]

print("\n[Classification Report — Best XGBoost]")
print(classification_report(y_val, y_pred_xgb,
                             target_names=["Normal (0)", "High price (1)"]))

# -----------------------------------------------------------------------------
# FEATURE IMPORTANCE — XGBoost
# -----------------------------------------------------------------------------

feature_names = (
    list(preprocessor.named_transformers_["cat"]
         .get_feature_names_out() if hasattr(
             preprocessor.named_transformers_["cat"], "get_feature_names_out"
         ) else CAT_FEATURES)
    + NUM_FEATURES
)
# Use raw feature names since we transformed manually
raw_names = CAT_FEATURES + NUM_FEATURES
importance = pd.Series(
    best_xgb["model"].feature_importances_,
    index=raw_names
).sort_values(ascending=False)

print("\n[XGBoost Feature Importance]")
print(importance.round(4).to_string())

# -----------------------------------------------------------------------------
# MODEL COMPARISON SUMMARY
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)
logistic_auc = 0.6051   # from Module 04
best_dt_auc  = best_dt["auc"]
best_rf_auc  = best_rf_row["auc"]
best_xgb_auc = best_xgb["auc"]

models  = ["Logistic Reg\n(Module 03)", "Decision Tree\n(best depth)", "Random Forest\n(n=best)", "XGBoost\n(best config)"]
aucs    = [logistic_auc, best_dt_auc, best_rf_auc, best_xgb_auc]
colors  = ["#9E9E9E", "#2196F3", "#4CAF50", "#FF9800"]

print(f"  {'Model':<25} {'AUC':>8}")
print("  " + "-" * 35)
for m, a in zip(models, aucs):
    print(f"  {m.replace(chr(10), ' '):<25} {a:>8.4f}")

# -----------------------------------------------------------------------------
# VISUALIZATIONS
# -----------------------------------------------------------------------------

print("\n[INFO] Generating plots ...")

# --- Plot 1: AUC comparison across models ---
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar([m.replace("\n", "\n") for m in models], aucs, color=colors, edgecolor="white", width=0.5)
ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Random (AUC=0.5)")
ax.axhline(logistic_auc, color="gray", linestyle=":", linewidth=1.2, label=f"Logistic baseline ({logistic_auc:.3f})")
for bar, auc in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{auc:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("ROC-AUC (Validation)", fontsize=12)
ax.set_title("Model Comparison — ROC-AUC on Temporal Validation Set (2019–2021)", fontsize=12)
ax.set_ylim(0.4, 1.05)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "18_model_comparison_auc.png"), dpi=150)
plt.close()
print("[INFO] Saved: 18_model_comparison_auc.png")

# --- Plot 2: Decision tree AUC vs depth ---
dt_df = pd.DataFrame(dt_results)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(dt_df["depth"], dt_df["auc"], marker="o", color="#2196F3", linewidth=2)
ax.set_xlabel("max_depth (None = unlimited)", fontsize=12)
ax.set_ylabel("Val ROC-AUC", fontsize=12)
ax.set_title("Decision Tree — AUC vs max_depth", fontsize=12)
ax.axhline(logistic_auc, color="gray", linestyle="--", linewidth=1, label=f"Logistic baseline ({logistic_auc:.3f})")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "19_dt_depth_vs_auc.png"), dpi=150)
plt.close()
print("[INFO] Saved: 19_dt_depth_vs_auc.png")

# --- Plot 3: XGBoost feature importance ---
fig, ax = plt.subplots(figsize=(8, 4))
importance.sort_values().plot(kind="barh", ax=ax, color="#FF9800")
ax.set_xlabel("Feature Importance (gain)", fontsize=12)
ax.set_title("XGBoost Feature Importance", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "20_xgb_feature_importance.png"), dpi=150)
plt.close()
print("[INFO] Saved: 20_xgb_feature_importance.png")

print("\n[INFO] Module 06 complete.")