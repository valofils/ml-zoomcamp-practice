# =============================================================================
# MODULE 10 — BentoML Service Definition (BentoML 1.4+ compatible)
# Serves the neural network via a BentoML REST API.
#
# Run from the project root:
#   bentoml serve 10-serving/service.py:svc --reload
#
# Test with PowerShell:
#   Invoke-WebRequest -Uri "http://localhost:3000/predict" `
#     -Method POST `
#     -ContentType "application/json" `
#     -Body '{"adm0_name":"Rwanda","cur_name":"RWF","adm1_name":"Kigali City","mp_year":2020,"mp_month":6}'
#
# Or open http://localhost:3000 for the Swagger UI
# =============================================================================

import os
import pickle
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import bentoml

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PKL_FILE = os.path.join(os.path.dirname(__file__), "preprocessor.pkl")

# -----------------------------------------------------------------------------
# SCHEMAS
# -----------------------------------------------------------------------------

class PriceAlertRequest(BaseModel):
    adm0_name : str = Field(..., example="Rwanda",       description="Country name")
    cur_name  : str = Field(..., example="RWF",          description="Currency code")
    adm1_name : str = Field("Unknown", example="Kigali City", description="Sub-national region")
    mp_year   : int = Field(..., example=2020,           description="Year")
    mp_month  : int = Field(..., example=6,              description="Month 1-12", ge=1, le=12)


class PriceAlertResponse(BaseModel):
    high_price       : int   = Field(..., description="1=high price, 0=normal")
    high_price_proba : float = Field(..., description="Probability of high price")
    alert            : str   = Field(..., description="Human-readable alert")

# -----------------------------------------------------------------------------
# SERVICE
# -----------------------------------------------------------------------------

@bentoml.service(
    name="maize-price-alert",
    resources={"cpu": "2"},
    traffic={"timeout": 30},
)
class MaizePriceAlertService:

    def __init__(self):
        self._nn = bentoml.tensorflow.load_model("maize_price_nn:latest")
        with open(PKL_FILE, "rb") as f:
            art = pickle.load(f)
        self._enc      = art["encoder"]
        self._cat_cols = art["cat_cols"]
        self._year_min = int(art["year_min"])

    def _build_input(self, req: PriceAlertRequest):
        month      = req.mp_month
        year_norm  = float(req.mp_year - self._year_min) / 31.0
        cat_raw    = pd.DataFrame([{c: getattr(req, c.replace("adm0", "adm0").replace("cur", "cur").replace("adm1", "adm1"))
                                    for c in self._cat_cols}])
        cat_raw    = pd.DataFrame([{
            "adm0_name": req.adm0_name,
            "cur_name":  req.cur_name,
            "adm1_name": req.adm1_name,
        }])
        cat_enc = self._enc.transform(cat_raw[self._cat_cols]).astype(np.int32) + 1
        num_arr = np.array([[
            year_norm,
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
        ]], dtype=np.float32)
        return {
            "cat_adm0_name": cat_enc[:, 0].reshape(-1, 1),
            "cat_cur_name":  cat_enc[:, 1].reshape(-1, 1),
            "cat_adm1_name": cat_enc[:, 2].reshape(-1, 1),
            "num_input":     num_arr,
        }

    @bentoml.api
    def predict(self, request: PriceAlertRequest) -> PriceAlertResponse:
        inputs = self._build_input(request)
        proba  = float(self._nn(inputs, training=False).numpy()[0][0])
        pred   = int(proba >= 0.5)
        return PriceAlertResponse(
            high_price       = pred,
            high_price_proba = round(proba, 4),
            alert            = "HIGH PRICE ALERT" if pred == 1 else "Normal price level",
        )


svc = MaizePriceAlertService