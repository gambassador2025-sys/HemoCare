FROM python:3.10-slim
WORKDIR /app

# install system deps (if any)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# collectstatic if you use staticfiles (not needed now)
EXPOSE 8000
CMD ["gunicorn", "hbppg_backend.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "2"]
