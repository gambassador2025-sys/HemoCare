# predictor/predictor.py
import os
import glob
import joblib
import numpy as np
from typing import List, Optional, Union
from tensorflow.keras.models import load_model
import threading
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MAX_NN_MODELS = int(os.environ.get("MAX_NN_MODELS", "3"))
USE_NN = os.environ.get("USE_NN", "1") not in ("0", "false", "False")

class HbPredictor:
    def __init__(self, model_dir: str = "./models"):
        self.model_dir = model_dir
        self._lock = threading.Lock()
        self.X_scaler = None
        self.labelencoder = None
        self.rf_model = None
        self.gb_model = None
        self.nn_models = []          # loaded keras models
        self._nn_paths = []
        self.nn_meta = None          # will hold {'y_mean':.., 'y_std':.., 'model_paths':[...] } if present
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

        # try to load nn meta (y_mean/y_std)
        meta_path = os.path.join(self.model_dir, "nn_ensemble.meta.joblib")
        if os.path.exists(meta_path):
            try:
                meta = joblib.load(meta_path)
                if isinstance(meta, dict):
                    self.nn_meta = meta
                    logger.info(f"Loaded NN ensemble meta from {meta_path} (keys: {list(meta.keys())})")
            except Exception as e:
                logger.warning(f"Failed to load nn_ensemble.meta.joblib: {e}")

    def _discover_nn_paths(self):
        # prefer nn_ensemble.meta joblib 'model_paths' if present
        if self.nn_meta and "model_paths" in self.nn_meta:
            candidate_paths = []
            for p in self.nn_meta.get("model_paths", []):
                # meta might store relative paths; normalize to model_dir
                candidate = os.path.join(self.model_dir, os.path.basename(p))
                if os.path.exists(candidate):
                    candidate_paths.append(candidate)
            if candidate_paths:
                self._nn_paths = candidate_paths
                logger.info(f"Using {len(self._nn_paths)} NN paths from meta file")
                return

        # fallback: find .h5 files
        pattern = os.path.join(self.model_dir, "nn_*_bag*.h5")
        files = sorted(glob.glob(pattern))
        if not files:
            files = sorted(glob.glob(os.path.join(self.model_dir, "*.h5")))
        self._nn_paths = files
        logger.info(f"Discovered {len(self._nn_paths)} .h5 NN files by glob")

    def _select_nn_to_load(self) -> List[str]:
        if not self._nn_paths:
            return []
        n_total = len(self._nn_paths)
        if n_total <= MAX_NN_MODELS:
            return self._nn_paths
        indices = np.linspace(0, n_total - 1, num=MAX_NN_MODELS, dtype=int)
        selected = [self._nn_paths[i] for i in indices]
        logger.info(f"Selecting {len(selected)}/{n_total} NN models to load: {[os.path.basename(p) for p in selected]}")
        return selected

    def _load_nn_models(self):
        selected = self._select_nn_to_load()
        loaded = []
        for p in selected:
            try:
                m = load_model(p, compile=False)
                loaded.append(m)
                logger.info(f"Loaded NN model {os.path.basename(p)}")
            except Exception as e:
                logger.warning(f"Failed to load NN model {p}: {e}")
        self.nn_models = loaded
        logger.info(f"Total NN models loaded: {len(self.nn_models)}")

    def _map_gender(self, gender):
        v = str(gender).strip().lower()
        if v in ("m", "male"):
            return "male"
        if v in ("f", "female"):
            return "female"
        return v

    def _unscale_nn_pred(self, val: float) -> float:
        """If nn_meta contains y_mean and y_std, convert model output back to original units."""
        if not self.nn_meta:
            return val
        try:
            y_mean = float(self.nn_meta.get("y_mean", 0.0))
            y_std = float(self.nn_meta.get("y_std", 1.0))
            return float(val * y_std + y_mean)
        except Exception:
            return val

    def predict(self, red: float, ir: float, gender: str, age: float, debug: bool = False) -> Union[float, dict]:
        """
        If debug=True, returns a dict with per-model preds:
        { "rf":..., "gb":..., "nn": [...], "final": ... }
        Otherwise returns a float final prediction.
        """
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

            if self.X_scaler is not None:
                try:
                    x_scaled = self.X_scaler.transform(x)
                except Exception:
                    x_scaled = x
            else:
                x_scaled = x

            preds = []
            breakdown = {"rf": None, "gb": None, "nn": []}

            if self.rf_model is not None:
                try:
                    rf_pred = self.rf_model.predict(x_scaled)
                    rf_val = float(np.ravel(rf_pred)[0])
                    breakdown["rf"] = rf_val
                    preds.append(rf_val)
                except Exception as e:
                    logger.warning(f"RF predict error: {e}")

            if self.gb_model is not None:
                try:
                    gb_pred = self.gb_model.predict(x_scaled)
                    gb_val = float(np.ravel(gb_pred)[0])
                    breakdown["gb"] = gb_val
                    preds.append(gb_val)
                except Exception as e:
                    logger.warning(f"GB predict error: {e}")

            nn_preds = []
            for m in self.nn_models:
                try:
                    raw = m.predict(x_scaled, verbose=0).flatten()[0]
                    unscaled = self._unscale_nn_pred(float(raw))
                    nn_preds.append(unscaled)
                except Exception as e:
                    logger.warning(f"NN predict failed: {e}")
            if nn_preds:
                breakdown["nn"] = nn_preds
                preds.append(float(np.mean(nn_preds)))

            if not preds:
                raise RuntimeError("No models available to make prediction. Ensure rf/gb or NN exist and loadable.")

            final = float(np.mean(preds))

            if debug:
                return {**breakdown, "final": final}
            return final
