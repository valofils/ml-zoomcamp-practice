# =============================================================================
# MODULE 09 — Test the Lambda handler
#
# Call the handler directly (no Docker or AWS needed):
#   python 09-serverless/test.py
#
# Or send the request to a running container or a deployed Function URL:
#   python 09-serverless/test.py --url http://localhost:9000/2015-03-31/functions/function/invocations
# =============================================================================

import os
import sys
import json
import argparse

REQUEST = {
    "adm0_name": "Rwanda",
    "cur_name" : "RWF",
    "adm1_name": "Kigali City",
    "mp_year"  : 2020,
    "mp_month" : 6,
}

parser = argparse.ArgumentParser()
parser.add_argument("--url", help="Lambda invocation URL (emulator or Function URL)")
args = parser.parse_args()

if args.url:
    import requests
    result = requests.post(args.url, json=REQUEST, timeout=30).json()
else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from lambda_handler import lambda_handler
    result = lambda_handler(REQUEST, None)

print(json.dumps(result, indent=2))
