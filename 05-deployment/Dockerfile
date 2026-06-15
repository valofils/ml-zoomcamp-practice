FROM python:3.11-slim

WORKDIR /app

# Install only what the service needs (not the full ML stack)
RUN pip install --no-cache-dir \
    fastapi==0.137.1 \
    uvicorn==0.49.0 \
    scikit-learn==1.9.0 \
    numpy==2.4.6 \
    pandas==2.2.3 \
    pydantic>=2.9.0

# Copy service files and the serialized model
COPY app.py predict.py ./
COPY model/ ./model/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
