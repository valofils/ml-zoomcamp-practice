# =============================================================================
# MODULE 09 — Serverless Deployment (AWS Lambda)
# Serves the logistic regression pipeline from Module 05 as a Lambda function.
# The model only needs scikit-learn, which keeps the image small and cold
# starts fast (TensorFlow would add over 1 GB to the image).
#
# Works with direct invocation (event = request JSON) and with API Gateway /
# Lambda Function URLs (request JSON in event["body"]).
#
# Test locally (no Docker needed):
#   python 09-serverless/test.py
# =============================================================================

import os
import sys
import json
import pickle

# Reuse the feature builder from Module 05 so Lambda and FastAPI build
# features the same way. In the container image predict.py sits next to
# this file; locally it is imported from 05-deployment.
HERE       = os.path.dirname(os.path.abspath(__file__))
DEPLOY_DIR = os.path.join(HERE, "..", "05-deployment")
if os.path.isdir(DEPLOY_DIR):
    sys.path.insert(0, DEPLOY_DIR)

from predict import build_features

MODEL_FILE = os.environ.get(
    "MODEL_FILE", os.path.join(DEPLOY_DIR, "model", "logistic_pipeline.pkl")
)

# -----------------------------------------------------------------------------
# LOAD MODEL ONCE PER CONTAINER (reused across warm invocations)
# -----------------------------------------------------------------------------

with open(MODEL_FILE, "rb") as f:
    _artifact = pickle.load(f)

PIPELINE = _artifact["pipeline"]
YEAR_MIN = _artifact["year_min"]

# -----------------------------------------------------------------------------
# REQUEST PARSING AND VALIDATION
# Same rules as the FastAPI schema in 05-deployment/app.py
# -----------------------------------------------------------------------------

def parse_event(event):
    """Return the request dict from a direct or API Gateway/Function URL event."""
    if isinstance(event, dict) and "body" in event:
        body = event["body"] or "{}"
        if event.get("isBase64Encoded"):
            import base64
            body = base64.b64decode(body).decode("utf-8")
        return json.loads(body) if isinstance(body, str) else body
    return event


def validate(request):
    """Return (observation, None) if valid, or (None, error message)."""
    if not isinstance(request, dict):
        return None, "Request body must be a JSON object"

    for field in ["adm0_name", "cur_name"]:
        if not isinstance(request.get(field), str) or not request[field]:
            return None, f"'{field}' is required and must be a non-empty string"

    adm1_name = request.get("adm1_name", "Unknown")
    if not isinstance(adm1_name, str):
        return None, "'adm1_name' must be a string"

    for field in ["mp_year", "mp_month"]:
        value = request.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            return None, f"'{field}' is required and must be an integer"

    if not 1 <= request["mp_month"] <= 12:
        return None, "'mp_month' must be between 1 and 12"

    return {
        "adm0_name": request["adm0_name"],
        "cur_name" : request["cur_name"],
        "adm1_name": adm1_name,
        "mp_year"  : request["mp_year"],
        "mp_month" : request["mp_month"],
    }, None


def response(status_code, body):
    return {
        "statusCode": status_code,
        "headers"   : {"Content-Type": "application/json"},
        "body"      : json.dumps(body),
    }

# -----------------------------------------------------------------------------
# HANDLER
# -----------------------------------------------------------------------------

def lambda_handler(event, context):
    try:
        request = parse_event(event)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return response(400, {"error": "Request body is not valid JSON"})

    observation, error = validate(request)
    if error:
        return response(400, {"error": error})

    features = build_features(observation, YEAR_MIN)
    proba    = float(PIPELINE.predict_proba(features)[0][1])
    pred     = int(PIPELINE.predict(features)[0])

    return response(200, {
        "high_price"      : pred,
        "high_price_proba": round(proba, 4),
        "alert"           : "HIGH PRICE ALERT" if pred == 1 else "Normal price level",
    })
