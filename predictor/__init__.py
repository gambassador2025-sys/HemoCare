# top of backend/main.py — MUST come before importing tensorflow
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # force CPU-only; prevents TF from trying to init CUDA
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # optional: reduce TF logs
default_app_config = 'predictor.apps.PredictorConfig'
