# =============================================================================
# MODULE 02 — Exploratory Data Analysis (EDA)
# Dataset : WFP Global Food Prices Database
# Strategy: Single commodity (Maize - Retail) across all countries
#           Target = log(mp_price) to handle currency scale differences
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "wfpvam_foodprices.csv")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# LOAD & INITIAL CLEAN
# -----------------------------------------------------------------------------

print("[INFO] Loading data ...")
df = pd.read_csv(DATA_FILE, low_memory=False)
print(f"[INFO] Raw shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Drop 100% missing column and numeric ID surrogates
drop_cols = [
    "mp_commoditysource",
    "adm0_id", "adm1_id", "mkt_id", "cm_id", "cur_id", "pt_id", "um_id",
]
df.drop(columns=drop_cols, inplace=True)

# Keep KG/L units and Retail price type only
df = df[df["um_name"].isin(["KG", "L"])].copy()
df = df[df["pt_name"] == "Retail"].copy()

# Fill missing sub-national region
df["adm1_name"] = df["adm1_name"].fillna("Unknown")

# Remove zero prices
df = df[df["mp_price"] > 0].copy()

print(f"[INFO] After base filters: {df.shape[0]:,} rows, {df['adm0_name'].nunique()} countries")

# -----------------------------------------------------------------------------
# FILTER TO MAIZE (all variants grouped together)
# cm_name contains: "Maize - Retail", "Maize (white) - Retail", etc.
# -----------------------------------------------------------------------------

maize_mask = df["cm_name"].str.lower().str.startswith("maize")
df_maize = df[maize_mask].copy()
print(f"[INFO] Maize records: {df_maize.shape[0]:,} rows")
print(f"[INFO] Maize variants found:")
print(df_maize["cm_name"].value_counts().to_string())

# Normalize label: group all maize variants under one name
df_maize["commodity"] = "Maize"

# -----------------------------------------------------------------------------
# REMOVE EXTREME PRICE OUTLIERS per country
# Use 1st and 99th percentile within each country to stay conservative
# -----------------------------------------------------------------------------

def remove_outliers_by_group(data, group_col, price_col, low=0.01, high=0.99):
    filtered = []
    for name, grp in data.groupby(group_col):
        lo = grp[price_col].quantile(low)
        hi = grp[price_col].quantile(high)
        filtered.append(grp[(grp[price_col] >= lo) & (grp[price_col] <= hi)])
    return pd.concat(filtered, ignore_index=True)

df_maize = remove_outliers_by_group(df_maize, "adm0_name", "mp_price")
print(f"[INFO] After per-country outlier removal: {df_maize.shape[0]:,} rows")

# -----------------------------------------------------------------------------
# LOG-TRANSFORM THE TARGET
# log(price) compresses the scale across currencies and reduces skewness
# This is the variable our model will predict
# -----------------------------------------------------------------------------

df_maize["log_price"] = np.log(df_maize["mp_price"])

# -----------------------------------------------------------------------------
# BUILD DATE COLUMN
# -----------------------------------------------------------------------------

df_maize["date"] = pd.to_datetime(
    df_maize["mp_year"].astype(str) + "-" +
    df_maize["mp_month"].astype(str).str.zfill(2) + "-01"
)

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("MAIZE DATASET SUMMARY")
print("=" * 60)
print(f"  Rows       : {df_maize.shape[0]:,}")
print(f"  Countries  : {df_maize['adm0_name'].nunique()}")
print(f"  Markets    : {df_maize['mkt_name'].nunique()}")
print(f"  Currencies : {df_maize['cur_name'].nunique()}")
print(f"  Date range : {df_maize['date'].min().date()} → {df_maize['date'].max().date()}")
print(f"\n  mp_price (local currency):")
print(df_maize["mp_price"].describe().round(3).to_string())
print(f"\n  log_price (target):")
print(df_maize["log_price"].describe().round(3).to_string())

# Top 10 countries by record count
print("\n[Top 10 countries by record count]")
print(df_maize["adm0_name"].value_counts().head(10).to_string())

# -----------------------------------------------------------------------------
# VISUALIZATIONS
# -----------------------------------------------------------------------------

print("\n[INFO] Generating plots ...")

# --- Plot 1: Raw price distribution vs log price ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(df_maize["mp_price"], bins=80, color="#2196F3", edgecolor="white", linewidth=0.3)
axes[0].set_title("Raw Price (local currency)", fontsize=12)
axes[0].set_xlabel("Price")
axes[0].set_ylabel("Frequency")
axes[0].set_yscale("log")

axes[1].hist(df_maize["log_price"], bins=80, color="#4CAF50", edgecolor="white", linewidth=0.3)
axes[1].set_title("log(Price) — model target", fontsize=12)
axes[1].set_xlabel("log(Price)")
axes[1].set_ylabel("Frequency")

plt.suptitle("Maize Price Distribution — Before and After Log Transform", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "01_price_distribution.png"), dpi=150)
plt.close()
print("[INFO] Saved: 01_price_distribution.png")

# --- Plot 2: log_price by country (top 15 by count) ---
top15 = df_maize["adm0_name"].value_counts().head(15).index.tolist()
df_top15 = df_maize[df_maize["adm0_name"].isin(top15)]
order = (
    df_top15.groupby("adm0_name")["log_price"]
    .median()
    .sort_values()
    .index.tolist()
)
fig, ax = plt.subplots(figsize=(11, 6))
sns.boxplot(data=df_top15, x="log_price", y="adm0_name", order=order, ax=ax, color="#FF9800")
ax.set_title("log(Maize Price) Distribution by Country (Top 15)", fontsize=13)
ax.set_xlabel("log(Price in local currency)")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "02_logprice_by_country.png"), dpi=150)
plt.close()
print("[INFO] Saved: 02_logprice_by_country.png")

# --- Plot 3: log_price trend over time (top 5 countries) ---
top5 = df_maize["adm0_name"].value_counts().head(5).index.tolist()
df_trend = (
    df_maize[df_maize["adm0_name"].isin(top5)]
    .groupby(["date", "adm0_name"])["log_price"]
    .median()
    .reset_index()
)
fig, ax = plt.subplots(figsize=(12, 5))
for country in top5:
    subset = df_trend[df_trend["adm0_name"] == country]
    ax.plot(subset["date"], subset["log_price"], label=country, linewidth=1.5)
ax.set_title("Maize log(Price) Trend — Top 5 Countries by Record Count", fontsize=13)
ax.set_xlabel("Date")
ax.set_ylabel("log(Price)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "03_logprice_trend.png"), dpi=150)
plt.close()
print("[INFO] Saved: 03_logprice_trend.png")

# --- Plot 4: log_price by month (seasonality check) ---
monthly = df_maize.groupby("mp_month")["log_price"].median()
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(monthly.index, monthly.values, color="#9C27B0", edgecolor="white")
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"])
ax.set_title("Median log(Maize Price) by Month — Seasonality Check", fontsize=13)
ax.set_xlabel("Month")
ax.set_ylabel("Median log(Price)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "04_seasonality.png"), dpi=150)
plt.close()
print("[INFO] Saved: 04_seasonality.png")

# --- Plot 5: log_price vs year (trend check) ---
yearly = df_maize.groupby("mp_year")["log_price"].median()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(yearly.index, yearly.values, marker="o", linewidth=2, color="#F44336", markersize=4)
ax.set_title("Median log(Maize Price) by Year — Long-term Trend", fontsize=13)
ax.set_xlabel("Year")
ax.set_ylabel("Median log(Price)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "05_yearly_trend.png"), dpi=150)
plt.close()
print("[INFO] Saved: 05_yearly_trend.png")

# -----------------------------------------------------------------------------
# SAVE CLEAN MAIZE DATASET FOR NEXT MODULES
# -----------------------------------------------------------------------------

keep_cols = [
    "adm0_name", "adm1_name", "mkt_name", "cur_name",
    "mp_month", "mp_year", "mp_price", "log_price", "date"
]
df_out = df_maize[keep_cols].copy()
out_path = os.path.join(os.path.dirname(__file__), "..", "data", "wfp_maize_clean.csv")
df_out.to_csv(out_path, index=False)
print(f"\n[INFO] Clean maize dataset saved to: {out_path}")
print(f"[INFO] Shape: {df_out.shape[0]:,} rows × {df_out.shape[1]} columns")
print("\n[INFO] EDA complete. Plots saved to: 02-regression/plots/")