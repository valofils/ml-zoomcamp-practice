# =============================================================================
# MODULE 10 — BentoML: Save Model to Model Store
# Stores the .keras file in a generic BentoML model (bentoml.tensorflow is
# deprecated and does not round-trip Keras 3 models).
# Preprocessor saved as a plain pickle file alongside the service.
#
# Run:
#   python 10-serving/train_and_save.py
# =============================================================================

import os
import shutil
import pickle
import numpy as np
import pandas as pd
import bentoml

from sklearn.preprocessing import OrdinalEncoder

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow import keras

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------

DATA_FILE  = os.path.join(os.path.dirname(__file__), "..", "data", "wfp_maize_clean.csv")
KERAS_FILE = os.path.join(os.path.dirname(__file__), "..", "08-deep-learning", "model", "nn_model.keras")
PKL_FILE   = os.path.join(os.path.dirname(__file__), "preprocessor.pkl")

# -----------------------------------------------------------------------------
# PREPARE ENCODER
# -----------------------------------------------------------------------------

print("[INFO] Loading data ...")
df = pd.read_csv(DATA_FILE, parse_dates=["date"])

country_median   = df.groupby("adm0_name")["log_price"].median().rename("country_median")
df               = df.join(country_median, on="adm0_name")
df["high_price"] = (df["log_price"] > df["country_median"]).astype(int)

CAT_COLS = ["adm0_name", "cur_name", "adm1_name"]
YEAR_MIN = int(df["mp_year"].min())
YEAR_MAX = int(df["mp_year"].max())

train = df[df["mp_year"] < 2019].copy()
enc   = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int32)
enc.fit(train[CAT_COLS])
print(f"[INFO] Encoder fitted on {len(train):,} training rows.")

# Save preprocessor as plain pickle (loaded directly by service.py)
artifact = {"encoder": enc, "cat_cols": CAT_COLS, "year_min": YEAR_MIN, "year_max": YEAR_MAX}
with open(PKL_FILE, "wb") as f:
    pickle.dump(artifact, f)
print(f"[INFO] Preprocessor saved to: {PKL_FILE}")

# -----------------------------------------------------------------------------
# LOAD KERAS MODEL AND SAVE IT TO THE BENTOML MODEL STORE
# -----------------------------------------------------------------------------

print(f"[INFO] Loading Keras model from: {KERAS_FILE}")
nn_model = keras.models.load_model(KERAS_FILE)
print(f"[INFO] Model loaded. Parameters: {nn_model.count_params():,}")

with bentoml.models.create(
    "maize_price_nn",
    metadata={
        "val_roc_auc"  : 0.7448,
        "val_f1"       : 0.7938,
        "dataset"      : "WFP Global Food Prices (maize, 1992-2021)",
        "trained_by"   : "valofils",
        "architecture" : "Embedding + Dense + BatchNorm + Dropout",
    },
) as bento_model:
    shutil.copy(KERAS_FILE, bento_model.path_of("nn_model.keras"))
print(f"[INFO] Model saved to BentoML store: {bento_model.tag}")

# -----------------------------------------------------------------------------
# SMOKE TEST
# -----------------------------------------------------------------------------

print("\n[INFO] Running smoke test ...")
loaded = keras.models.load_model(bentoml.models.get("maize_price_nn:latest").path_of("nn_model.keras"))

with open(PKL_FILE, "rb") as f:
    art = pickle.load(f)

sample_cat = art["encoder"].transform(
    pd.DataFrame([{"adm0_name": "Rwanda", "cur_name": "RWF", "adm1_name": "Kigali City"}])[art["cat_cols"]]
).astype(np.int32) + 1

test_input = {
    "cat_adm0_name": sample_cat[:, 0].reshape(-1, 1),
    "cat_cur_name":  sample_cat[:, 1].reshape(-1, 1),
    "cat_adm1_name": sample_cat[:, 2].reshape(-1, 1),
    "num_input":     np.array([[
        (2020 - YEAR_MIN) / (YEAR_MAX - YEAR_MIN),
        np.sin(2 * np.pi * 3 / 12),
        np.cos(2 * np.pi * 3 / 12),
    ]], dtype=np.float32),
}
proba = float(loaded(test_input, training=False).numpy()[0][0])
print(f"[INFO] Rwanda 2020-03: proba={proba:.4f}  → {'HIGH PRICE' if proba >= 0.5 else 'Normal'}")

print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"  BentoML model tag : {bento_model.tag}")
print(f"  Preprocessor pkl  : {PKL_FILE}")
print(f"\n  Next step:")
print(f"  bentoml serve 10-serving/service.py:svc --reload")
print("\n[INFO] train_and_save.py complete.")