# predictor/predictor.py
import os
import glob
import joblib
import numpy as np
from typing import List, Optional
from tensorflow.keras.models import load_model
import threading
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Config via env
MAX_NN_MODELS = int(os.environ.get("MAX_NN_MODELS", "3"))
USE_NN = os.environ.get("USE_NN", "1") not in ("0", "false", "False")

class HbPredictor:
    """
    Loads small joblib models and (optionally) a limited number of Keras .h5 models
    at startup. Models are reused across requests to avoid repeated loading and OOM.
    """
    def __init__(self, model_dir: str = "./models"):
        self.model_dir = model_dir
        self._lock = threading.Lock()
        self.X_scaler = None
        self.labelencoder = None
        self.rf_model = None
        self.gb_model = None
        self.nn_models = []  # loaded keras models
        self._nn_paths = []
        self._load_small_models()
        if USE_NN:
            self._discover_nn_paths()
            self._load_nn_models()
        else:
            logger.info("USE_NN=0 -> skipping NN model load (using rf/gb only if present)")

    def _load_small_models(self):
        # load X_scaler
        for p in ("X_scaler.save", "X_scaler.joblib", "X_scaler.pkl"):
            path = os.path.join(self.model_dir, p)
            if os.path.exists(path):
                try:
                    self.X_scaler = joblib.load(path)
                    logger.info(f"Loaded X_scaler from {path}")
                except Exception as e:
                    logger.warning(f"Failed to load scaler {path}: {e}")
                break

        # label encoder
        for p in ("labelencoder.save", "labelencoder.joblib", "labelencoder.pkl"):
            path = os.path.join(self.model_dir, p)
            if os.path.exists(path):
                try:
                    self.labelencoder = joblib.load(path)
                    logger.info(f"Loaded labelencoder from {path}")
                except Exception as e:
                    logger.warning(f"Failed to load labelencoder {path}: {e}")
                break

        # rf_model
        for p in ("rf_model.joblib", "rf_model.save", "rf_model.pkl"):
            path = os.path.join(self.model_dir, p)
            if os.path.exists(path):
                try:
                    self.rf_model = joblib.load(path)
                    logger.info(f"Loaded RF model from {path}")
                except Exception as e:
                    logger.warning(f"Failed to load RF model {path}: {e}")
                break

        # gb_model
        for p in ("gb_model.joblib", "gb_model.save", "gb_model.pkl"):
            path = os.path.join(self.model_dir, p)
            if os.path.exists(path):
                try:
                    self.gb_model = joblib.load(path)
                    logger.info(f"Loaded GB model from {path}")
                except Exception as e:
                    logger.warning(f"Failed to load GB model {path}: {e}")
                break

    def _discover_nn_paths(self):
        # prefer nn_ensemble.meta.joblib if present (training script saved it)
        meta_path = os.path.join(self.model_dir, "nn_ensemble.meta.joblib")
        if os.path.exists(meta_path):
            try:
                meta = joblib.load(meta_path)
                paths = meta.get("model_paths") if isinstance(meta, dict) else None
                if paths:
                    # normalize to absolute paths in models dir
                    self._nn_paths = [os.path.join(self.model_dir, os.path.basename(p)) for p in paths]
                    # only include those that actually exist
                    self._nn_paths = [p for p in self._nn_paths if os.path.exists(p)]
                    logger.info(f"nn_ensemble.meta.joblib provided {len(self._nn_paths)} paths")
                else:
                    logger.info("nn_ensemble.meta.joblib present but no 'model_paths' key")
            except Exception as e:
                logger.warning(f"Failed to read nn_ensemble.meta.joblib: {e}")

        # fallback: glob for .h5 files
        if not self._nn_paths:
            pattern = os.path.join(self.model_dir, "nn_*_bag*.h5")
            files = sorted(glob.glob(pattern))
            if not files:
                files = sorted(glob.glob(os.path.join(self.model_dir, "*.h5")))
            self._nn_paths = files
            logger.info(f"Discovered {len(self._nn_paths)} .h5 models by glob")

    def _select_nn_to_load(self) -> List[str]:
        # Select up to MAX_NN_MODELS evenly across available files if many
        if not self._nn_paths:
            return []
        n_total = len(self._nn_paths)
        if n_total <= MAX_NN_MODELS:
            return self._nn_paths
        # choose indices spread across folds: simple downsampling
        indices = np.linspace(0, n_total - 1, num=MAX_NN_MODELS, dtype=int)
        selected = [self._nn_paths[i] for i in indices]
        logger.info(f"Selecting {len(selected)}/{n_total} NN models to load: {[os.path.basename(p) for p in selected]}")
        return selected

    def _load_nn_models(self):
        selected = self._select_nn_to_load()
        loaded = []
        for path in selected:
            try:
                m = load_model(path, compile=False)
                loaded.append(m)
                logger.info(f"Loaded NN model {os.path.basename(path)}")
            except Exception as e:
                logger.warning(f"Failed to load NN model {path}: {e}")
        self.nn_models = loaded
        logger.info(f"Total NN models loaded: {len(self.nn_models)}")

    def _map_gender(self, gender):
        v = str(gender).strip().lower()
        if v in ("m", "male"):
            return "male"
        if v in ("f", "female"):
            return "female"
        return v

    def predict(self, red: float, ir: float, gender: str, age: float) -> float:
        with self._lock:
            g = self._map_gender(gender)
            if self.labelencoder is not None:
                try:
                    gender_enc = int(self.labelencoder.transform([g])[0])
                except Exception:
                    try:
                        gender_enc = int(self.labelencoder.transform([g.capitalize()])[0])
                    except Exception:
                        gender_enc = 1 if g == "male" else 0
            else:
                gender_enc = 1 if g == "male" else 0

            x = np.array([[red, ir, gender_enc, age]], dtype=np.float32)

            # scale if scaler present
            if self.X_scaler is not None:
                try:
                    x_scaled = self.X_scaler.transform(x)
                except Exception:
                    x_scaled = x
            else:
                x_scaled = x

            preds = []

            # classical models
            if self.rf_model is not None:
                try:
                    rf_pred = self.rf_model.predict(x_scaled)
                    preds.append(float(np.ravel(rf_pred)[0]))
                except Exception as e:
                    logger.warning(f"RF prediction failed: {e}")
            if self.gb_model is not None:
                try:
                    gb_pred = self.gb_model.predict(x_scaled)
                    preds.append(float(np.ravel(gb_pred)[0]))
                except Exception as e:
                    logger.warning(f"GB prediction failed: {e}")

            # NN ensemble if loaded
            nn_preds = []
            for m in self.nn_models:
                try:
                    p = m.predict(x_scaled, verbose=0).flatten()[0]
                    nn_preds.append(float(p))
                except Exception as e:
                    logger.warning(f"NN model predict error: {e}")
                    continue
            if nn_preds:
                preds.append(float(np.mean(nn_preds)))

            if not preds:
                # final fallback: raise explicit error so caller can respond 500
                raise RuntimeError("No models available to make prediction. Ensure rf/gb or NN models exist and are loadable.")

            final = float(np.mean(preds))
            return final
