# predictor/predictor.py
import os
import glob
import joblib
import numpy as np
from typing import Optional
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import threading

class HbPredictor:
    """
    Lightweight predictor that:
      - Loads small joblib models/scalers at init (X scaler, label encoder, rf, gb).
      - Does NOT keep all Keras .h5 models in memory. Instead it loads each NN .h5
        file one-by-one during predict(), gets predictions, then immediately deletes
        it and clears Keras session to free memory.
    """
    def __init__(self, model_dir: str = "./models"):
        self.model_dir = model_dir
        self._lock = threading.Lock()
        # placeholders
        self.X_scaler = None
        self.labelencoder = None
        self.rf_model = None
        self.gb_model = None
        self.nn_files = []  # list of paths to .h5 NN models
        self._load_small_models()

    def _load_small_models(self):
        # load joblib scalers / classical models if present
        # try common filenames based on your training code
        x_scaler_paths = [
            os.path.join(self.model_dir, "X_scaler.save"),
            os.path.join(self.model_dir, "X_scaler.joblib"),
            os.path.join(self.model_dir, "X_scaler.pkl"),
        ]
        for p in x_scaler_paths:
            if os.path.exists(p):
                self.X_scaler = joblib.load(p)
                break

        le_paths = [
            os.path.join(self.model_dir, "labelencoder.save"),
            os.path.join(self.model_dir, "labelencoder.joblib"),
            os.path.join(self.model_dir, "labelencoder.pkl"),
        ]
        for p in le_paths:
            if os.path.exists(p):
                self.labelencoder = joblib.load(p)
                break

        rf_paths = [
            os.path.join(self.model_dir, "rf_model.joblib"),
            os.path.join(self.model_dir, "rf_model.save"),
            os.path.join(self.model_dir, "rf_model.pkl"),
        ]
        for p in rf_paths:
            if os.path.exists(p):
                self.rf_model = joblib.load(p)
                break

        gb_paths = [
            os.path.join(self.model_dir, "gb_model.joblib"),
            os.path.join(self.model_dir, "gb_model.save"),
            os.path.join(self.model_dir, "gb_model.pkl"),
        ]
        for p in gb_paths:
            if os.path.exists(p):
                self.gb_model = joblib.load(p)
                break

        # find NN files (h5). We'll lazy-load them during predict
        pattern = os.path.join(self.model_dir, "nn_*_bag*.h5")
        files = sorted(glob.glob(pattern))
        # also accept any .h5 in model_dir as fallback
        if not files:
            files = sorted(glob.glob(os.path.join(self.model_dir, "*.h5")))
        self.nn_files = files

    def _map_gender(self, gender):
        # Accept many variants; return encoded string expected by labelencoder or manual mapping.
        v = str(gender).strip().lower()
        if v in ("m", "male"):
            return "male"
        if v in ("f", "female"):
            return "female"
        # fallback: just return as is
        return v

    def predict(self, red: float, ir: float, gender: str, age: float) -> float:
        """
        Returns a single float predicted hemoglobin.
        This method is thread-safe.
        """
        with self._lock:
            # Build input array
            g = self._map_gender(gender)
            # if labelencoder is present, transform gender to encoded value
            if self.labelencoder is not None:
                try:
                    gender_enc = int(self.labelencoder.transform([g])[0])
                except Exception:
                    # if transform fails, try lower/upper variants
                    try:
                        gender_enc = int(self.labelencoder.transform([g.capitalize()])[0])
                    except Exception:
                        # fallback: try manual mapping to 0/1
                        gender_enc = 1 if g == "male" else 0
            else:
                gender_enc = 1 if g == "male" else 0

            x = np.array([[red, ir, gender_enc, age]], dtype=np.float32)

            # apply X scaler if available
            if self.X_scaler is not None:
                try:
                    x_scaled = self.X_scaler.transform(x)
                except Exception:
                    # if scaler expects different shape, try fallback
                    x_scaled = x
            else:
                x_scaled = x

            preds = []

            # RF prediction
            if self.rf_model is not None:
                try:
                    rf_pred = self.rf_model.predict(x_scaled if hasattr(self.rf_model, "predict") else x)
                    preds.append(float(rf_pred.flatten()[0]))
                except Exception:
                    pass

            # GB prediction
            if self.gb_model is not None:
                try:
                    gb_pred = self.gb_model.predict(x_scaled if hasattr(self.gb_model, "predict") else x)
                    preds.append(float(gb_pred.flatten()[0]))
                except Exception:
                    pass

            # NN ensemble predictions: load each .h5 model one by one, predict, then clear.
            nn_preds = []
            for nn_path in self.nn_files:
                try:
                    # load model with compile=False to reduce cost if you don't need training
                    model = load_model(nn_path, compile=False)
                    p = model.predict(x_scaled, verbose=0).flatten()[0]
                    nn_preds.append(float(p))
                    # free memory used by model
                    del model
                    K.clear_session()
                except Exception:
                    # if one model fails to load/predict, skip it
                    continue

            if nn_preds:
                # average bagged NN predictions
                nn_mean = float(np.mean(nn_preds))
                preds.append(nn_mean)

            if not preds:
                raise RuntimeError("No models available to make prediction.")

            # Final ensemble: simple average of available model predictions
            final = float(np.mean(preds))
            return final
