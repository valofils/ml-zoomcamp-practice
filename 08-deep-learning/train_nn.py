# =============================================================================
# MODULE 08 — Neural Networks & Deep Learning
# Task   : Same binary classification (high_price) as Modules 03 and 06
# Model  : Feedforward neural network with Keras/TensorFlow
# Goal   : Demonstrate the deep learning workflow and compare with XGBoost
#
# Architecture:
#   - OrdinalEncoder for categoricals (integer codes fed to Embedding layers)
#   - Embedding layers for adm0_name, cur_name, adm1_name
#   - Dense layers with BatchNormalization and Dropout
#   - Binary output with sigmoid activation
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score, f1_score, classification_report

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TF info/warning logs
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(f"[INFO] TensorFlow version: {tf.__version__}")

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "wfp_maize_clean.csv")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# LOAD & PREPARE
# -----------------------------------------------------------------------------

print("[INFO] Loading data ...")
df = pd.read_csv(DATA_FILE, parse_dates=["date"])

country_median   = df.groupby("adm0_name")["log_price"].median().rename("country_median")
df               = df.join(country_median, on="adm0_name")
df["high_price"] = (df["log_price"] > df["country_median"]).astype(int)
df["year_norm"]  = (df["mp_year"] - df["mp_year"].min()) / (df["mp_year"].max() - df["mp_year"].min())
df["month_sin"]  = np.sin(2 * np.pi * df["mp_month"] / 12)
df["month_cos"]  = np.cos(2 * np.pi * df["mp_month"] / 12)

CAT_COLS = ["adm0_name", "cur_name", "adm1_name"]
NUM_COLS = ["year_norm", "month_sin", "month_cos"]
TARGET   = "high_price"

# Temporal split
train = df[df["mp_year"] < 2019].copy()
val   = df[df["mp_year"] >= 2019].copy()
print(f"[INFO] Train: {len(train):,} | Val: {len(val):,}")

# -----------------------------------------------------------------------------
# ENCODE CATEGORICALS
# Embedding layers need integer indices (0, 1, 2, ...) not OHE or strings
# -----------------------------------------------------------------------------

enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                     dtype=np.int32)
enc.fit(train[CAT_COLS])

def encode(data):
    cat = enc.transform(data[CAT_COLS]).astype(np.int32)
    # Shift by 1 so unknown (-1) becomes 0 and valid codes start at 1
    cat = cat + 1
    num = data[NUM_COLS].values.astype(np.float32)
    return cat, num

cat_train, num_train = encode(train)
cat_val,   num_val   = encode(val)
y_train = train[TARGET].values.astype(np.float32)
y_val   = val[TARGET].values.astype(np.float32)

# Vocabulary sizes (+ 1 for unknown token, + 1 for 1-based indexing)
vocab_sizes = [int(len(c)) + 2 for c in enc.categories_]
print(f"[INFO] Vocabulary sizes: {dict(zip(CAT_COLS, vocab_sizes))}")

# -----------------------------------------------------------------------------
# BUILD MODEL
# Functional API — separate inputs for each categorical + numeric block
# -----------------------------------------------------------------------------

def build_model(vocab_sizes, num_dim, embed_dim=8, dropout_rate=0.3):
    """
    vocab_sizes : list of int, one per categorical feature
    num_dim     : number of numerical features
    embed_dim   : embedding dimension for each categorical
    dropout_rate: dropout applied after each dense layer
    """
    cat_inputs  = []
    cat_embeds  = []

    for i, (name, vocab) in enumerate(zip(CAT_COLS, vocab_sizes)):
        inp = keras.Input(shape=(1,), dtype="int32", name=f"cat_{name}")
        emb = layers.Embedding(input_dim=vocab, output_dim=embed_dim, name=f"emb_{name}")(inp)
        emb = layers.Flatten(name=f"flat_{name}")(emb)
        cat_inputs.append(inp)
        cat_embeds.append(emb)

    num_input = keras.Input(shape=(num_dim,), dtype="float32", name="num_input")
    x = layers.Concatenate(name="concat")(cat_embeds + [num_input])

    x = layers.Dense(128, activation="relu", name="dense_1")(x)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(dropout_rate, name="drop_1")(x)

    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Dropout(dropout_rate, name="drop_2")(x)

    x = layers.Dense(32, activation="relu", name="dense_3")(x)
    x = layers.Dropout(dropout_rate / 2, name="drop_3")(x)

    output = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=cat_inputs + [num_input], outputs=output)
    return model


model = build_model(vocab_sizes, num_dim=len(NUM_COLS))
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")],
)
model.summary()

# -----------------------------------------------------------------------------
# PREPARE INPUTS AS DICT (Functional API expects named inputs)
# -----------------------------------------------------------------------------

def make_inputs(cat_array, num_array):
    return {
        f"cat_{name}": cat_array[:, i].reshape(-1, 1)
        for i, name in enumerate(CAT_COLS)
    } | {"num_input": num_array}

train_inputs = make_inputs(cat_train, num_train)
val_inputs   = make_inputs(cat_val,   num_val)

# -----------------------------------------------------------------------------
# TRAIN WITH EARLY STOPPING
# -----------------------------------------------------------------------------

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_auc", patience=5, mode="max",
        restore_best_weights=True, verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc", factor=0.5, patience=3, mode="max",
        min_lr=1e-5, verbose=1,
    ),
]

print("\n[INFO] Training neural network ...")
history = model.fit(
    train_inputs, y_train,
    validation_data=(val_inputs, y_val),
    epochs=50,
    batch_size=2048,
    callbacks=callbacks,
    verbose=1,
)

# -----------------------------------------------------------------------------
# EVALUATE
# -----------------------------------------------------------------------------

y_proba = model.predict(val_inputs, verbose=0).flatten()
y_pred  = (y_proba >= 0.5).astype(int)

auc = roc_auc_score(y_val, y_proba)
f1  = f1_score(y_val, y_pred)

print("\n" + "=" * 55)
print("NEURAL NETWORK — FINAL METRICS")
print("=" * 55)
print(f"  Val ROC-AUC : {auc:.4f}")
print(f"  Val F1      : {f1:.4f}")
print()
print(classification_report(y_val, y_pred,
                             target_names=["Normal (0)", "High price (1)"]))

# Comparison
print("=" * 55)
print("COMPARISON WITH PREVIOUS MODELS")
print("=" * 55)
comparison = {
    "Logistic Regression": 0.6051,
    "Decision Tree":       0.6558,
    "Random Forest":       0.6494,
    "XGBoost":             0.6896,
    "Neural Network":      auc,
}
for name, score in comparison.items():
    bar = "█" * int(score * 40)
    print(f"  {name:<22} {score:.4f}  {bar}")

# -----------------------------------------------------------------------------
# SAVE MODEL
# -----------------------------------------------------------------------------

model_path = os.path.join(MODEL_DIR, "nn_model.keras")
model.save(model_path)
print(f"\n[INFO] Model saved to: {model_path}")

# -----------------------------------------------------------------------------
# VISUALIZATIONS
# -----------------------------------------------------------------------------

print("\n[INFO] Generating plots ...")
epochs_ran = len(history.history["auc"])

# --- Plot 1: Training curves ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["loss"],     label="Train loss", linewidth=2)
axes[0].plot(history.history["val_loss"], label="Val loss",   linewidth=2)
axes[0].set_title("Loss", fontsize=12)
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history.history["auc"],     label="Train AUC", linewidth=2)
axes[1].plot(history.history["val_auc"], label="Val AUC",   linewidth=2)
axes[1].set_title("ROC-AUC", fontsize=12)
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.suptitle("Neural Network Training Curves", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "23_nn_training_curves.png"), dpi=150)
plt.close()
print("[INFO] Saved: 23_nn_training_curves.png")

# --- Plot 2: Model comparison ---
models  = list(comparison.keys())
scores  = list(comparison.values())
colors  = ["#9E9E9E", "#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(models, scores, color=colors, edgecolor="white", width=0.5)
ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Random (0.50)")
for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{score:.3f}", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("ROC-AUC (Temporal Val 2019–2021)", fontsize=12)
ax.set_title("All Models — ROC-AUC Comparison", fontsize=13)
ax.set_ylim(0.4, 1.0)
ax.legend()
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "24_all_models_comparison.png"), dpi=150)
plt.close()
print("[INFO] Saved: 24_all_models_comparison.png")

print("\n[INFO] Module 08 complete.")