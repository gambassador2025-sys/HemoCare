FROM python:3.10-slim
ENV CUDA_VISIBLE_DEVICES="-1"
ENV TF_CPP_MIN_LOG_LEVEL="2"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["gunicorn", "hbppg_backend.wsgi:application", "--bind", "0.0.0.0:8000"]
