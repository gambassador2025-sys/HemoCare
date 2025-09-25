FROM python:3.10-slim

# Force CPU-only (prevents cuInit errors) and quiet TF logs
ENV CUDA_VISIBLE_DEVICES="-1"
ENV TF_CPP_MIN_LOG_LEVEL="2"

# Limit how many NN models to load by default
ENV MAX_NN_MODELS="3"
# Toggle NN usage: set USE_NN="0" to skip loading .h5 models and only use rf/gb
ENV USE_NN="1"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "hbppg_backend.wsgi:application", "--bind", "0.0.0.0:8000"]
