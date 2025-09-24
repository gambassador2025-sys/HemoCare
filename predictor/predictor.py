"""
HbPredictor loader and wrapper.
Expects a `models/` directory next to manage.py containing:
 - X_scaler.save
 - labelencoder.save
 - rf_model.joblib
 - gb_model.joblib
 - nn_ensemble.meta.joblib (optional) with keys: model_paths, y_mean, y_std
 - and the nn models referenced in model_paths (HDF5 .h5 or TF SavedModel)
"""
import os
import joblib
import numpy as np

# Keras import delayed to avoid heavy import on management commands that don't need prediction
def _load_keras():
    # import lazily
    from tensorflow.keras.models import load_model
    return load_model

class HbPredictor:
    def __init__(self, model_dir=None):
        if model_dir is None:
            # default: models directory at project root (same place as manage.py)
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(project_root, '..', 'models')
            model_dir = os.path.abspath(model_dir)
        self.model_dir = model_dir

        # load required artifacts
        self.X_scaler = joblib.load(os.path.join(self.model_dir, 'X_scaler.save'))
        self.labelencoder = joblib.load(os.path.join(self.model_dir, 'labelencoder.save'))
        self.rf = joblib.load(os.path.join(self.model_dir, 'rf_model.joblib'))
        self.gb = joblib.load(os.path.join(self.model_dir, 'gb_model.joblib'))

        # Optional: NN ensemble metadata
        self.nn_models = []
        self.y_mean = 0.0
        self.y_std = 1.0
        nn_meta_path = os.path.join(self.model_dir, 'nn_ensemble.meta.joblib')
        if os.path.exists(nn_meta_path):
            meta = joblib.load(nn_meta_path)
            self.y_mean = float(meta.get('y_mean', 0.0))
            self.y_std = float(meta.get('y_std', 1.0))
            model_paths = meta.get('model_paths', []) or []
            load_model = _load_keras()
            for p in model_paths:
                p_abs = os.path.join(self.model_dir, os.path.basename(p)) if not os.path.isabs(p) else p
                # allow both relative names or absolute
                self.nn_models.append(load_model(p_abs, compile=False))

    def predict(self, red, ir, gender, age):
        """
        Predict a single sample. `gender` must be normalized string matching label encoder labels
        returned by PredictRequestSerializer (i.e., 'male' or 'female').
        """
        # labelencoder expects array-like
        try:
            gender_enc = int(self.labelencoder.transform([gender])[0])
        except Exception as e:
            # fallback mapping if transform fails (but this is unlikely if you saved the same encoder)
            gender_enc = 1 if str(gender).lower().startswith('m') else 0

        x = np.array([[red, ir, gender_enc, age]], dtype=np.float32)
        x_s = self.X_scaler.transform(x)

        rf_pred = float(self.rf.predict(x_s)[0])
        gb_pred = float(self.gb.predict(x_s)[0])

        nn_preds = []
        for m in self.nn_models:
            p = m.predict(x_s, verbose=0).flatten()[0]
            # convert back to original scale if saved as scaled target
            p = p * self.y_std + self.y_mean
            nn_preds.append(float(p))

        nn_mean = float(np.mean(nn_preds)) if len(nn_preds) > 0 else rf_pred
        ensemble = (rf_pred + gb_pred + nn_mean) / 3.0
        return ensemble
