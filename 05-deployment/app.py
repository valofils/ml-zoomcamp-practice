# =============================================================================
# MODULE 05 — FastAPI Prediction Service
# Serves the maize high-price classification model via a REST API.
#
# Run with:
#   uvicorn 05-deployment.app:app --reload
# Or from inside the 05-deployment folder:
#   uvicorn app:app --reload
#
# Test with:
#   curl -X POST http://localhost:8000/predict \
#     -H "Content-Type: application/json" \
#     -d '{"adm0_name":"Rwanda","cur_name":"RWF","adm1_name":"Kigali City","mp_year":2020,"mp_month":6}'
# =============================================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from predict import predict

# -----------------------------------------------------------------------------
# APP
# -----------------------------------------------------------------------------

app = FastAPI(
    title="WFP Maize Price Alert API",
    description=(
        "Predicts whether a maize retail market is in a high-price state "
        "relative to its country's historical median. "
        "Based on WFP Global Food Prices data (1992–2021)."
    ),
    version="1.0.0",
)

# -----------------------------------------------------------------------------
# REQUEST / RESPONSE SCHEMAS
# Pydantic models validate the request body automatically.
# FastAPI generates OpenAPI docs from these schemas.
# -----------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    adm0_name : str  = Field(..., example="Rwanda",     description="Country name")
    cur_name  : str  = Field(..., example="RWF",        description="Currency code")
    adm1_name : str  = Field("Unknown", example="Kigali City", description="Sub-national region")
    mp_year   : int  = Field(..., example=2020,         description="Year (1992–2021)")
    mp_month  : int  = Field(..., example=6,            description="Month (1–12)", ge=1, le=12)


class PredictionResponse(BaseModel):
    high_price       : int   = Field(..., description="1 = high price, 0 = normal")
    high_price_proba : float = Field(..., description="Probability of high price (0–1)")
    alert            : str   = Field(..., description="Human-readable alert label")

# -----------------------------------------------------------------------------
# ENDPOINTS
# -----------------------------------------------------------------------------

@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "service": "WFP Maize Price Alert API", "version": "1.0.0"}


@app.post("/predict", response_model=PredictionResponse, summary="Predict high-price alert")
def predict_endpoint(request: PredictionRequest):
    """
    Accepts a market observation and returns a binary high-price prediction
    along with the model's confidence probability.
    """
    try:
        observation = request.model_dump()
        result = predict(observation)
        return result
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model file not found. Run 05-deployment/train.py first.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
