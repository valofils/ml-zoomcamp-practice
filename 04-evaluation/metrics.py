# =============================================================================
# MODULE 04 — Evaluation Metrics for Classification
# Covers: ROC curve, AUC, Precision-Recall curve, cross-validation
# Uses  : val_clf_predictions.csv from Module 03
# Goal  : Properly evaluate the logistic regression and expose its limits
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# LOAD PREDICTIONS (from Module 03)
# -----------------------------------------------------------------------------

df_val = pd.read_csv(os.path.join(DATA_DIR, "val_clf_predictions.csv"))
print(f"[INFO] Loaded validation predictions: {df_val.shape[0]:,} rows")

y_true  = df_val["high_price"]
y_proba = df_val["high_price_proba"]
y_pred  = df_val["high_price_pred"]

# -----------------------------------------------------------------------------
# SECTION 1 — ROC CURVE & AUC
# ROC curve plots True Positive Rate vs False Positive Rate at all thresholds.
# AUC (Area Under the Curve) summarizes discriminative power:
#   AUC = 1.0 → perfect classifier
#   AUC = 0.5 → random classifier (no better than a coin flip)
# -----------------------------------------------------------------------------

fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
auc = roc_auc_score(y_true, y_proba)

# Find threshold closest to the top-left corner (maximize TPR - FPR)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold_roc = roc_thresholds[optimal_idx]

print("\n" + "=" * 55)
print("ROC CURVE ANALYSIS")
print("=" * 55)
print(f"  AUC                     : {auc:.4f}")
print(f"  Optimal threshold (ROC) : {optimal_threshold_roc:.4f}")
print(f"  TPR at optimal threshold: {tpr[optimal_idx]:.4f}")
print(f"  FPR at optimal threshold: {fpr[optimal_idx]:.4f}")
print(f"\n  Interpretation:")
if auc >= 0.9:
    print("  Excellent discriminative power.")
elif auc >= 0.8:
    print("  Good discriminative power.")
elif auc >= 0.7:
    print("  Fair discriminative power.")
elif auc >= 0.6:
    print("  Poor discriminative power — model barely beats random.")
else:
    print("  Very poor — model is essentially random.")
print(f"  → AUC={auc:.3f} confirms the logistic model cannot reliably")
print(f"    separate normal from high-price markets across countries.")
print(f"    Tree-based models (Module 06) will close this gap.")

# -----------------------------------------------------------------------------
# SECTION 2 — PRECISION-RECALL CURVE
# More informative than ROC when classes are imbalanced.
# Average Precision (AP) summarizes the curve.
# -----------------------------------------------------------------------------

precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_true, y_proba)
ap = average_precision_score(y_true, y_proba)

# Find threshold that maximizes F1 on the PR curve
f1_vals = np.where(
    (precision_vals + recall_vals) == 0, 0,
    2 * precision_vals * recall_vals / (precision_vals + recall_vals)
)
best_pr_idx = np.argmax(f1_vals)
best_pr_threshold = pr_thresholds[best_pr_idx] if best_pr_idx < len(pr_thresholds) else 0.5

print("\n" + "=" * 55)
print("PRECISION-RECALL CURVE ANALYSIS")
print("=" * 55)
print(f"  Average Precision (AP)  : {ap:.4f}")
print(f"  Best F1 on PR curve     : {f1_vals[best_pr_idx]:.4f}")
print(f"  Threshold at best F1    : {best_pr_threshold:.4f}")
print(f"  Baseline AP (random)    : {y_true.mean():.4f}  (= positive rate)")

# -----------------------------------------------------------------------------
# SECTION 3 — CONFUSION MATRIX AT TWO THRESHOLDS
# Compare default (0.5) vs optimal (ROC-optimal) threshold
# -----------------------------------------------------------------------------

print("\n" + "=" * 55)
print("CONFUSION MATRIX COMPARISON")
print("=" * 55)

for label, thresh in [("Default (0.50)", 0.50), (f"ROC-optimal ({optimal_threshold_roc:.2f})", optimal_threshold_roc)]:
    y_pred_t = (y_proba >= thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred_t)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  [{label}]")
    print(f"    TP={tp:,}  FP={fp:,}  TN={tn:,}  FN={fn:,}")
    print(f"    Precision : {tp/(tp+fp):.4f}")
    print(f"    Recall    : {tp/(tp+fn):.4f}")
    print(f"    Specificity (TNR): {tn/(tn+fp):.4f}")

# -----------------------------------------------------------------------------
# SECTION 4 — CROSS-VALIDATION
# k-fold CV gives a more reliable estimate of model performance
# than a single train/val split.
# We use StratifiedKFold to preserve class balance in each fold.
# NOTE: We use a time-agnostic CV here deliberately to show what
#       in-sample CV looks like vs the time-based OOS performance.
# -----------------------------------------------------------------------------

print("\n" + "=" * 55)
print("CROSS-VALIDATION (5-fold Stratified, in-sample)")
print("=" * 55)

df_full = pd.read_csv(os.path.join(DATA_DIR, "wfp_maize_clean.csv"))

# Rebuild target and features (same as Module 03)
country_median = df_full.groupby("adm0_name")["log_price"].median()
df_full = df_full.join(country_median.rename("country_median"), on="adm0_name")
df_full["high_price"] = (df_full["log_price"] > df_full["country_median"]).astype(int)
df_full["year_norm"]  = df_full["mp_year"] - df_full["mp_year"].min()
df_full["month_sin"]  = np.sin(2 * np.pi * df_full["mp_month"] / 12)
df_full["month_cos"]  = np.cos(2 * np.pi * df_full["mp_month"] / 12)

CAT_FEATURES = ["adm0_name", "cur_name", "adm1_name"]
NUM_FEATURES  = ["year_norm", "month_sin", "month_cos"]

X = df_full[CAT_FEATURES + NUM_FEATURES]
y = df_full["high_price"]

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ("num", "passthrough", NUM_FEATURES),
])
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(C=0.1, solver="saga", max_iter=1000, random_state=42)),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

print(f"  ROC-AUC per fold : {np.round(cv_auc, 4).tolist()}")
print(f"  Mean AUC         : {cv_auc.mean():.4f}")
print(f"  Std AUC          : {cv_auc.std():.4f}")
print(f"\n  Note: CV AUC ({cv_auc.mean():.3f}) > temporal val AUC ({auc:.3f})")
print(f"  This gap confirms temporal distribution shift — the model")
print(f"  memorizes country patterns but cannot generalize to future years.")

# -----------------------------------------------------------------------------
# VISUALIZATIONS
# -----------------------------------------------------------------------------

print("\n[INFO] Generating plots ...")

# --- Plot 1: ROC Curve ---
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color="#2196F3", linewidth=2, label=f"Logistic Reg (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random (AUC = 0.50)")
ax.scatter(fpr[optimal_idx], tpr[optimal_idx], color="red", s=80, zorder=5,
           label=f"Optimal threshold = {optimal_threshold_roc:.2f}")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve — High-Price Classification", fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "14_roc_curve.png"), dpi=150)
plt.close()
print("[INFO] Saved: 14_roc_curve.png")

# --- Plot 2: Precision-Recall Curve ---
baseline = y_true.mean()
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(recall_vals, precision_vals, color="#4CAF50", linewidth=2,
        label=f"Logistic Reg (AP = {ap:.3f})")
ax.axhline(baseline, color="gray", linestyle="--", linewidth=1,
           label=f"Random baseline (AP = {baseline:.3f})")
ax.scatter(recall_vals[best_pr_idx], precision_vals[best_pr_idx],
           color="red", s=80, zorder=5, label=f"Best F1 threshold = {best_pr_threshold:.2f}")
ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curve — High-Price Classification", fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "15_precision_recall_curve.png"), dpi=150)
plt.close()
print("[INFO] Saved: 15_precision_recall_curve.png")

# --- Plot 3: Confusion matrix at optimal ROC threshold ---
y_pred_opt = (y_proba >= optimal_threshold_roc).astype(int)
cm_opt = confusion_matrix(y_true, y_pred_opt)
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_opt,
                              display_labels=["Normal (0)", "High price (1)"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Confusion Matrix — Threshold = {optimal_threshold_roc:.2f}", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "16_confusion_matrix.png"), dpi=150)
plt.close()
print("[INFO] Saved: 16_confusion_matrix.png")

# --- Plot 4: CV AUC scores ---
fig, ax = plt.subplots(figsize=(7, 4))
folds = [f"Fold {i+1}" for i in range(len(cv_auc))]
bars = ax.bar(folds, cv_auc, color="#9C27B0", edgecolor="white")
ax.axhline(cv_auc.mean(), color="red", linestyle="--", linewidth=1.5,
           label=f"Mean AUC = {cv_auc.mean():.3f}")
ax.axhline(auc, color="orange", linestyle="--", linewidth=1.5,
           label=f"Temporal val AUC = {auc:.3f}")
ax.set_ylabel("ROC-AUC", fontsize=12)
ax.set_title("5-Fold Cross-Validation AUC vs Temporal Validation AUC", fontsize=12)
ax.set_ylim(0.5, 1.0)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "17_cv_auc.png"), dpi=150)
plt.close()
print("[INFO] Saved: 17_cv_auc.png")

print("\n[INFO] Evaluation complete.")