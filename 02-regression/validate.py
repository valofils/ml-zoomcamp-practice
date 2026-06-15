# =============================================================================
# MODULE 02 — Model Validation & Residual Analysis
# Loads val_predictions.csv produced by train.py
# Diagnoses model performance: residuals, bias by country, error over time
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# LOAD PREDICTIONS
# -----------------------------------------------------------------------------

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "val_predictions.csv"))
print(f"[INFO] Loaded validation predictions: {df.shape[0]:,} rows")

# Derived columns
df["abs_residual"] = df["residual"].abs()
df["price_pred"]   = np.exp(df["log_price_pred"])   # back to original scale
df["price_error"]  = (df["price_pred"] - df["mp_price"]).abs()

# -----------------------------------------------------------------------------
# GLOBAL METRICS
# -----------------------------------------------------------------------------

rmse     = np.sqrt((df["residual"] ** 2).mean())
mae      = df["abs_residual"].mean()
mean_res = df["residual"].mean()   # bias: positive = underpredicting

print("\n" + "=" * 55)
print("VALIDATION METRICS (log scale)")
print("=" * 55)
print(f"  RMSE          : {rmse:.4f}")
print(f"  MAE           : {mae:.4f}")
print(f"  Mean residual : {mean_res:.4f}  {'(model underpredicts)' if mean_res > 0 else '(model overpredicts)'}")
print(f"  Exp(RMSE)     : ×{np.exp(rmse):.2f}  (avg multiplicative error on price scale)")

# -----------------------------------------------------------------------------
# RMSE BY COUNTRY
# -----------------------------------------------------------------------------

country_rmse = (
    df.groupby("adm0_name")["residual"]
    .apply(lambda x: np.sqrt((x**2).mean()))
    .sort_values(ascending=False)
    .rename("rmse")
)
print(f"\n[RMSE by country — worst 10]")
print(country_rmse.head(10).round(4).to_string())
print(f"\n[RMSE by country — best 10]")
print(country_rmse.tail(10).round(4).to_string())

# -----------------------------------------------------------------------------
# RMSE BY YEAR
# -----------------------------------------------------------------------------

year_rmse = (
    df.groupby("mp_year")["residual"]
    .apply(lambda x: np.sqrt((x**2).mean()))
    .rename("rmse")
)
print(f"\n[RMSE by year]")
print(year_rmse.round(4).to_string())

# -----------------------------------------------------------------------------
# VISUALIZATIONS
# -----------------------------------------------------------------------------

print("\n[INFO] Generating validation plots ...")

# --- Plot 1: Predicted vs Actual (log scale) ---
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(df["log_price"], df["log_price_pred"],
           alpha=0.15, s=5, color="#2196F3", rasterized=True)
lims = [df["log_price"].min(), df["log_price"].max()]
ax.plot(lims, lims, color="red", linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("Actual log(Price)", fontsize=12)
ax.set_ylabel("Predicted log(Price)", fontsize=12)
ax.set_title("Predicted vs Actual — log(Maize Price)", fontsize=13)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "06_pred_vs_actual.png"), dpi=150)
plt.close()
print("[INFO] Saved: 06_pred_vs_actual.png")

# --- Plot 2: Residual distribution ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df["residual"], bins=100, color="#9C27B0", edgecolor="white", linewidth=0.3)
ax.axvline(0, color="red", linewidth=1.5, linestyle="--", label="Zero residual")
ax.axvline(mean_res, color="orange", linewidth=1.5, linestyle="--",
           label=f"Mean residual = {mean_res:.3f}")
ax.set_xlabel("Residual (actual − predicted)", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.set_title("Residual Distribution", fontsize=13)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "07_residual_distribution.png"), dpi=150)
plt.close()
print("[INFO] Saved: 07_residual_distribution.png")

# --- Plot 3: RMSE by country (top 15 worst) ---
worst15 = country_rmse.head(15).sort_values()
fig, ax = plt.subplots(figsize=(10, 5))
worst15.plot(kind="barh", ax=ax, color="#F44336")
ax.set_xlabel("RMSE (log scale)", fontsize=12)
ax.set_title("Top 15 Countries with Highest Validation RMSE", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "08_rmse_by_country.png"), dpi=150)
plt.close()
print("[INFO] Saved: 08_rmse_by_country.png")

# --- Plot 4: RMSE by year ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(year_rmse.index, year_rmse.values, marker="o", color="#FF9800",
        linewidth=2, markersize=6)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("RMSE (log scale)", fontsize=12)
ax.set_title("Validation RMSE by Year (2019–2021)", fontsize=13)
ax.set_xticks(year_rmse.index)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "09_rmse_by_year.png"), dpi=150)
plt.close()
print("[INFO] Saved: 09_rmse_by_year.png")

# --- Plot 5: Residuals vs predicted (heteroscedasticity check) ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df["log_price_pred"], df["residual"],
           alpha=0.1, s=5, color="#4CAF50", rasterized=True)
ax.axhline(0, color="red", linewidth=1.5, linestyle="--")
ax.set_xlabel("Predicted log(Price)", fontsize=12)
ax.set_ylabel("Residual", fontsize=12)
ax.set_title("Residuals vs Predicted — Heteroscedasticity Check", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "10_residuals_vs_predicted.png"), dpi=150)
plt.close()
print("[INFO] Saved: 10_residuals_vs_predicted.png")

# -----------------------------------------------------------------------------
# DIAGNOSIS SUMMARY
# -----------------------------------------------------------------------------

print("\n" + "=" * 55)
print("DIAGNOSIS")
print("=" * 55)
gap = rmse - np.sqrt((df["residual"]**2).mean())
high_rmse = country_rmse[country_rmse > 1.0]
print(f"  Countries with RMSE > 1.0 : {len(high_rmse)} → {high_rmse.index.tolist()}")
print(f"  Mean residual ({mean_res:.3f}) → model {'underpredicts' if mean_res > 0 else 'overpredicts'} on 2019–2021 data")
print(f"  Next step: tree-based models (Module 06) will capture")
print(f"             non-linear country × year interactions")
print("=" * 55)