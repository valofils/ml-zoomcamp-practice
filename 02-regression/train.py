# =============================================================================
# MODULE 02 — Linear Regression Training
# Target  : log_price (log of maize retail price in local currency)
# Features: country, currency, market region, year, month
# =============================================================================

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
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
# FEATURE ENGINEERING
# -----------------------------------------------------------------------------

# Time-based features
df["year_norm"] = df["mp_year"] - df["mp_year"].min()   # years since start
df["month_sin"] = np.sin(2 * np.pi * df["mp_month"] / 12)  # cyclic encoding
df["month_cos"] = np.cos(2 * np.pi * df["mp_month"] / 12)  # cyclic encoding

# Categorical features to encode
CAT_FEATURES = ["adm0_name", "cur_name", "adm1_name"]

# Numerical features
NUM_FEATURES = ["year_norm", "month_sin", "month_cos"]

# Target
TARGET = "log_price"

print(f"\n[Features]")
print(f"  Categorical : {CAT_FEATURES}")
print(f"  Numerical   : {NUM_FEATURES}")
print(f"  Target      : {TARGET}")

# -----------------------------------------------------------------------------
# TRAIN / VALIDATION SPLIT
# Use time-based split: train on data before 2019, validate on 2019–2021
# This respects the temporal nature of the data (no data leakage)
# -----------------------------------------------------------------------------

SPLIT_YEAR = 2019
train = df[df["mp_year"] < SPLIT_YEAR].copy()
val   = df[df["mp_year"] >= SPLIT_YEAR].copy()

print(f"\n[Split] Train: {len(train):,} rows ({train['mp_year'].min()}–{train['mp_year'].max()})")
print(f"[Split] Val  : {len(val):,} rows ({val['mp_year'].min()}–{val['mp_year'].max()})")

X_train = train[CAT_FEATURES + NUM_FEATURES]
y_train = train[TARGET]

X_val = val[CAT_FEATURES + NUM_FEATURES]
y_val = val[TARGET]

# -----------------------------------------------------------------------------
# PREPROCESSING PIPELINE
# OneHotEncoder for categoricals, passthrough for numericals
# handle_unknown="ignore" ensures unseen categories in val don't crash the model
# -----------------------------------------------------------------------------

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ("num", "passthrough", NUM_FEATURES),
])

# -----------------------------------------------------------------------------
# MODEL 1 — PLAIN LINEAR REGRESSION (baseline)
# -----------------------------------------------------------------------------

lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression()),
])

lr_pipeline.fit(X_train, y_train)

y_pred_train_lr = lr_pipeline.predict(X_train)
y_pred_val_lr   = lr_pipeline.predict(X_val)

rmse_train_lr = np.sqrt(mean_squared_error(y_train, y_pred_train_lr))
rmse_val_lr   = np.sqrt(mean_squared_error(y_val, y_pred_val_lr))
r2_val_lr     = r2_score(y_val, y_pred_val_lr)

print("\n" + "=" * 55)
print("MODEL 1 — Linear Regression (no regularization)")
print("=" * 55)
print(f"  Train RMSE : {rmse_train_lr:.4f}")
print(f"  Val   RMSE : {rmse_val_lr:.4f}")
print(f"  Val   R²   : {r2_val_lr:.4f}")

# -----------------------------------------------------------------------------
# MODEL 2 — RIDGE REGRESSION (L2 regularization)
# Ridge adds a penalty proportional to the sum of squared coefficients.
# This prevents overfitting when we have many one-hot-encoded categories.
# We try several alpha values and pick the best on validation set.
# -----------------------------------------------------------------------------

print("\n" + "=" * 55)
print("MODEL 2 — Ridge Regression (L2 regularization)")
print("=" * 55)
print(f"  {'Alpha':>10} | {'Train RMSE':>12} | {'Val RMSE':>10} | {'Val R²':>8}")
print("  " + "-" * 48)

best_alpha   = None
best_rmse    = float("inf")
best_ridge   = None

for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    ridge_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", Ridge(alpha=alpha)),
    ])
    ridge_pipeline.fit(X_train, y_train)

    rmse_tr = np.sqrt(mean_squared_error(y_train, ridge_pipeline.predict(X_train)))
    rmse_vl = np.sqrt(mean_squared_error(y_val,   ridge_pipeline.predict(X_val)))
    r2_vl   = r2_score(y_val, ridge_pipeline.predict(X_val))

    marker = " ◄ best" if rmse_vl < best_rmse else ""
    print(f"  {alpha:>10.2f} | {rmse_tr:>12.4f} | {rmse_vl:>10.4f} | {r2_vl:>8.4f}{marker}")

    if rmse_vl < best_rmse:
        best_rmse  = rmse_vl
        best_alpha = alpha
        best_ridge = ridge_pipeline

print(f"\n  Best alpha : {best_alpha}")
print(f"  Best val RMSE (Ridge): {best_rmse:.4f}")

# -----------------------------------------------------------------------------
# INTERPRET RMSE ON ORIGINAL SCALE
# Since target = log(price), RMSE is in log units.
# exp(RMSE) gives the multiplicative error factor on the original price scale.
# e.g. RMSE=0.5 → prices are off by a factor of exp(0.5) ≈ 1.65 on average
# -----------------------------------------------------------------------------

print("\n" + "=" * 55)
print("INTERPRETATION")
print("=" * 55)
print(f"  Best Ridge val RMSE (log scale) : {best_rmse:.4f}")
print(f"  Approx multiplicative error     : ×{np.exp(best_rmse):.2f}")
print(f"  Meaning: predicted price is off by factor ~{np.exp(best_rmse):.2f} on average")

# -----------------------------------------------------------------------------
# FEATURE IMPORTANCE — top OHE coefficients from best Ridge model
# -----------------------------------------------------------------------------

feature_names = (
    best_ridge.named_steps["preprocessor"]
    .named_transformers_["cat"]
    .get_feature_names_out(CAT_FEATURES)
    .tolist()
    + NUM_FEATURES
)
coefs = best_ridge.named_steps["model"].coef_

importance = pd.Series(coefs, index=feature_names).abs().sort_values(ascending=False)
print(f"\n[Top 15 features by |coefficient|]")
print(importance.head(15).round(4).to_string())

# -----------------------------------------------------------------------------
# SAVE BEST MODEL INFO FOR validate.py
# (we save predictions to CSV; model object saved in 05-deployment)
# -----------------------------------------------------------------------------

val_results = val[["adm0_name", "mp_year", "mp_month", "mp_price", "log_price"]].copy()
val_results["log_price_pred"] = best_ridge.predict(X_val)
val_results["residual"]       = val_results["log_price"] - val_results["log_price_pred"]

out_path = os.path.join(os.path.dirname(__file__), "..", "data", "val_predictions.csv")
val_results.to_csv(out_path, index=False)
print(f"\n[INFO] Validation predictions saved to: {out_path}")
print("[INFO] Training complete.")
