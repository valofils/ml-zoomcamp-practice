# =============================================================================
# MODULE 03 — Classification: Prediction & Threshold Analysis
# Loads val_clf_predictions.csv produced by train.py
# Analyzes probability threshold effects on precision/recall trade-off
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score,
)

# -----------------------------------------------------------------------------
# LOAD
# -----------------------------------------------------------------------------

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "val_clf_predictions.csv"))
print(f"[INFO] Loaded: {df.shape[0]:,} validation predictions")

y_true  = df["high_price"]
y_proba = df["high_price_proba"]

# -----------------------------------------------------------------------------
# BASELINE METRICS (default threshold = 0.5)
# -----------------------------------------------------------------------------

y_pred_default = (y_proba >= 0.5).astype(int)

print("\n" + "=" * 55)
print("METRICS AT DEFAULT THRESHOLD (0.50)")
print("=" * 55)
print(f"  Accuracy  : {accuracy_score(y_true, y_pred_default):.4f}")
print(f"  Precision : {precision_score(y_true, y_pred_default, zero_division=0):.4f}")
print(f"  Recall    : {recall_score(y_true, y_pred_default, zero_division=0):.4f}")
print(f"  F1        : {f1_score(y_true, y_pred_default, zero_division=0):.4f}")
print(f"  ROC-AUC   : {roc_auc_score(y_true, y_proba):.4f}")

# -----------------------------------------------------------------------------
# THRESHOLD SWEEP
# The default threshold of 0.5 is not always optimal.
# For a food security alert system:
#   - High recall = don't miss crises (preferred)
#   - High precision = fewer false alarms
# We sweep thresholds and report all metrics to let the analyst decide.
# -----------------------------------------------------------------------------

thresholds  = np.arange(0.10, 0.91, 0.05)
results     = []

for t in thresholds:
    y_pred = (y_proba >= t).astype(int)
    results.append({
        "threshold" : round(t, 2),
        "accuracy"  : accuracy_score(y_true, y_pred),
        "precision" : precision_score(y_true, y_pred, zero_division=0),
        "recall"    : recall_score(y_true, y_pred, zero_division=0),
        "f1"        : f1_score(y_true, y_pred, zero_division=0),
    })

results_df = pd.DataFrame(results)
best_f1_row = results_df.loc[results_df["f1"].idxmax()]
best_t      = best_f1_row["threshold"]

print("\n" + "=" * 65)
print("THRESHOLD SWEEP")
print("=" * 65)
print(f"  {'Threshold':>10} | {'Accuracy':>10} | {'Precision':>10} | {'Recall':>8} | {'F1':>8}")
print("  " + "-" * 58)
for _, row in results_df.iterrows():
    marker = " ◄" if row["threshold"] == best_t else ""
    print(f"  {row['threshold']:>10.2f} | {row['accuracy']:>10.4f} | "
          f"{row['precision']:>10.4f} | {row['recall']:>8.4f} | "
          f"{row['f1']:>8.4f}{marker}")

print(f"\n  Best threshold by F1 : {best_t}")
print(f"  Best F1              : {best_f1_row['f1']:.4f}")
print(f"  Precision at best T  : {best_f1_row['precision']:.4f}")
print(f"  Recall at best T     : {best_f1_row['recall']:.4f}")

# -----------------------------------------------------------------------------
# PERFORMANCE BY COUNTRY AT BEST THRESHOLD
# -----------------------------------------------------------------------------

y_pred_best = (y_proba >= best_t).astype(int)
df["pred_best"] = y_pred_best

country_metrics = []
for country, grp in df.groupby("adm0_name"):
    if len(grp) < 10:
        continue
    f1  = f1_score(grp["high_price"], grp["pred_best"], zero_division=0)
    rec = recall_score(grp["high_price"], grp["pred_best"], zero_division=0)
    country_metrics.append({"country": country, "f1": f1, "recall": rec, "n": len(grp)})

cm_df = pd.DataFrame(country_metrics).sort_values("f1")
print(f"\n[F1 by country — worst 8]")
print(cm_df.head(8)[["country", "f1", "recall", "n"]].to_string(index=False))
print(f"\n[F1 by country — best 8]")
print(cm_df.tail(8)[["country", "f1", "recall", "n"]].to_string(index=False))

# -----------------------------------------------------------------------------
# VISUALIZATIONS
# -----------------------------------------------------------------------------

print("\n[INFO] Generating plots ...")

# --- Plot 1: Precision / Recall / F1 vs Threshold ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(results_df["threshold"], results_df["precision"], label="Precision", linewidth=2, color="#2196F3")
ax.plot(results_df["threshold"], results_df["recall"],    label="Recall",    linewidth=2, color="#F44336")
ax.plot(results_df["threshold"], results_df["f1"],        label="F1",        linewidth=2, color="#4CAF50")
ax.axvline(best_t, color="gray", linestyle="--", linewidth=1.2, label=f"Best threshold = {best_t}")
ax.set_xlabel("Classification Threshold", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Precision / Recall / F1 vs Threshold — High-Price Alert Model", fontsize=12)
ax.legend()
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "11_threshold_sweep.png"), dpi=150)
plt.close()
print("[INFO] Saved: 11_threshold_sweep.png")

# --- Plot 2: Predicted probability distribution by true class ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df.loc[df["high_price"]==0, "high_price_proba"], bins=60,
        alpha=0.6, color="#2196F3", label="Actual: Normal (0)", density=True)
ax.hist(df.loc[df["high_price"]==1, "high_price_proba"], bins=60,
        alpha=0.6, color="#F44336", label="Actual: High price (1)", density=True)
ax.axvline(best_t, color="black", linestyle="--", linewidth=1.5, label=f"Best threshold = {best_t}")
ax.set_xlabel("Predicted Probability of High Price", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("Predicted Probability Distribution by True Class", fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "12_proba_distribution.png"), dpi=150)
plt.close()
print("[INFO] Saved: 12_proba_distribution.png")

# --- Plot 3: F1 by country (bottom 15) ---
worst15 = cm_df.head(15).sort_values("f1")
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(worst15["country"], worst15["f1"], color="#FF9800")
ax.set_xlabel("F1 Score", fontsize=12)
ax.set_title(f"Countries with Lowest F1 (threshold={best_t})", fontsize=12)
ax.set_xlim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "13_f1_by_country.png"), dpi=150)
plt.close()
print("[INFO] Saved: 13_f1_by_country.png")

print("\n[INFO] predict.py complete.")