# =============================================================================
# MODULE 07 — MLflow Model Registry
# Registers the best model (XGBoost) from the tracking server into the
# MLflow Model Registry and transitions it to "Production" stage.
#
# Run AFTER tracking.py:
#   python 07-mlflow/registry.py
# =============================================================================

import os
import mlflow
from mlflow.tracking import MlflowClient

# -----------------------------------------------------------------------------
# CONNECT TO THE SAME TRACKING SERVER
# -----------------------------------------------------------------------------

DB_PATH = os.path.join(os.path.dirname(__file__), "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath(DB_PATH)}")

client = MlflowClient()

EXPERIMENT_NAME = "wfp-maize-price-alert"
REGISTERED_NAME = "maize-price-alert-xgboost"

# -----------------------------------------------------------------------------
# FIND THE BEST RUN BY VAL_ROC_AUC
# -----------------------------------------------------------------------------

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found. Run tracking.py first.")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.val_roc_auc DESC"],
    max_results=10,
)

print(f"\n[All runs ranked by val_roc_auc]")
print(f"  {'Run name':<40} {'AUC':>8} {'F1':>8}")
print("  " + "-" * 58)
for run in runs:
    name = run.data.tags.get("mlflow.runName", run.info.run_id[:8])
    auc  = run.data.metrics.get("val_roc_auc", 0)
    f1   = run.data.metrics.get("val_f1", 0)
    print(f"  {name:<40} {auc:>8.4f} {f1:>8.4f}")

best_run = runs[0]
best_run_id   = best_run.info.run_id
best_run_name = best_run.data.tags.get("mlflow.runName", best_run_id[:8])
best_auc      = best_run.data.metrics.get("val_roc_auc", 0)

print(f"\n[Best run] {best_run_name}  (AUC={best_auc:.4f})")

# -----------------------------------------------------------------------------
# CHECK IF THIS RUN HAS A LOGGED MODEL ARTIFACT
# Only runs where we called mlflow.xgboost.log_model have an artifact
# -----------------------------------------------------------------------------

artifacts = [a.path for a in client.list_artifacts(best_run_id)]
print(f"[Artifacts in best run] {artifacts}")

if "model" not in artifacts:
    print(f"\n[WARNING] Best run '{best_run_name}' has no logged model artifact.")
    print("  Falling back to the XGBoost run which has a logged model.")
    for run in runs:
        name = run.data.tags.get("mlflow.runName", run.info.run_id[:8])
        if "xgboost" in name.lower():
            arts = [a.path for a in client.list_artifacts(run.info.run_id)]
            if "model" in arts:
                best_run     = run
                best_run_id  = run.info.run_id
                best_run_name = name
                best_auc     = run.data.metrics.get("val_roc_auc", 0)
                print(f"  Using: {best_run_name}  (AUC={best_auc:.4f})")
                break

# -----------------------------------------------------------------------------
# REGISTER THE MODEL
# -----------------------------------------------------------------------------

model_uri = f"runs:/{best_run_id}/model"
print(f"\n[INFO] Registering model from URI: {model_uri}")

result = mlflow.register_model(model_uri=model_uri, name=REGISTERED_NAME)
version = result.version

print(f"[INFO] Registered as '{REGISTERED_NAME}' version {version}")

# -----------------------------------------------------------------------------
# ADD DESCRIPTION AND TAGS
# -----------------------------------------------------------------------------

client.update_registered_model(
    name=REGISTERED_NAME,
    description=(
        "XGBoost classifier predicting whether a maize retail market is in a "
        "high-price state relative to the country's historical median. "
        "Trained on WFP Global Food Prices data (1992–2018), "
        "validated on 2019–2021."
    ),
)

client.update_model_version(
    name=REGISTERED_NAME,
    version=version,
    description=f"Best model from experiment '{EXPERIMENT_NAME}'. Val AUC={best_auc:.4f}.",
)

client.set_model_version_tag(REGISTERED_NAME, version, "val_roc_auc", str(round(best_auc, 4)))
client.set_model_version_tag(REGISTERED_NAME, version, "dataset", "WFP maize prices 1992-2021")
client.set_model_version_tag(REGISTERED_NAME, version, "trained_by", "valofils")

print(f"[INFO] Description and tags added to version {version}")

# -----------------------------------------------------------------------------
# SET ALIAS (replaces deprecated stage in MLflow 3.x)
# In MLflow 3.x, aliases replace the "Production/Staging" stage concept.
# -----------------------------------------------------------------------------

client.set_registered_model_alias(
    name=REGISTERED_NAME,
    alias="production",
    version=version,
)
print(f"[INFO] Alias 'production' set on version {version}")

# -----------------------------------------------------------------------------
# VERIFY
# -----------------------------------------------------------------------------

print("\n" + "=" * 55)
print("REGISTRY SUMMARY")
print("=" * 55)

reg_model = client.get_registered_model(REGISTERED_NAME)
print(f"  Name        : {reg_model.name}")
print(f"  Description : {reg_model.description[:80]}...")

versions = client.search_model_versions(f"name='{REGISTERED_NAME}'")
for v in versions:
    aliases = v.aliases if hasattr(v, "aliases") else []
    print(f"  Version {v.version}   tags={dict(v.tags)}  aliases={aliases}")

print(f"\n  Load production model with:")
print(f"  mlflow.pyfunc.load_model('models:/{REGISTERED_NAME}@production')")
print("\n[INFO] registry.py complete.")
