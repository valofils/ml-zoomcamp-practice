# AWS Lambda container image for the maize price alert model.
# Build from the project root (the image needs files from 05-deployment):
#   python 05-deployment/train.py
#   docker build -f 09-serverless/Dockerfile -t maize-price-lambda .
#   docker run -p 9000:8080 maize-price-lambda

FROM public.ecr.aws/lambda/python:3.13

# Same versions the model was trained with, so the pickle loads
RUN pip install --no-cache-dir \
    scikit-learn==1.9.0 \
    numpy==2.4.6 \
    pandas==2.2.3

COPY 05-deployment/predict.py 09-serverless/lambda_handler.py ${LAMBDA_TASK_ROOT}/
COPY 05-deployment/model/logistic_pipeline.pkl ${LAMBDA_TASK_ROOT}/model/

ENV MODEL_FILE=${LAMBDA_TASK_ROOT}/model/logistic_pipeline.pkl

CMD ["lambda_handler.lambda_handler"]
