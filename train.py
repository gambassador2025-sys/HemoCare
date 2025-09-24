"""
train.py

Adapted from your earlier script. Use to train models and save artifacts into ./models

Example:
    python train.py --csv "/path/to/Final Dataset Hb PPG.csv" --out models
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

parser = argparse.ArgumentParser()
parser.add_argument('--csv', required=True, help='Path to CSV dataset')
parser.add_argument('--out', default='models', help='Output models dir')
parser.add_argument('--random_state', type=int, default=42)
parser.add_argument('--nn_count', type=int, default=3, help='Number of NN models to train/save')
args = parser.parse_args()

OUT_DIR = args.out
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(args.csv)
df = df.rename(columns={
    'Red (a.u)': 'Red',
    'Infra Red (a.u)': 'IR',
    'Age (year)': 'Age',
    'Hemoglobin (g/dL)': 'Hemoglobin'
})
required_cols = {"Red","IR","Gender","Age","Hemoglobin"}
assert required_cols.issubset(df.columns), f"CSV must have columns: {required_cols}"

df = df.dropna(subset=list(required_cols)).reset_index(drop=True)
le = LabelEncoder()
df['Gender_enc'] = le.fit_transform(df['Gender'].astype(str))

X = df[['Red','IR','Gender_enc','Age']].values.astype(np.float32)
y = df['Hemoglobin'].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=args.random_state)

X_scaler = StandardScaler().fit(X_train)
X_train_s = X_scaler.transform(X_train)
X_test_s = X_scaler.transform(X_test)

# RandomForest & GradientBoosting
rf = RandomForestRegressor(n_estimators=200, random_state=args.random_state, n_jobs=-1)
gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=args.random_state)
rf.fit(X_train_s, y_train)
gb.fit(X_train_s, y_train)

joblib.dump(rf, os.path.join(OUT_DIR, 'rf_model.joblib'))
joblib.dump(gb, os.path.join(OUT_DIR, 'gb_model.joblib'))
joblib.dump(X_scaler, os.path.join(OUT_DIR, 'X_scaler.save'))
joblib.dump(le, os.path.join(OUT_DIR, 'labelencoder.save'))

# Train small ensemble of NNs and save
nn_paths = []
y_mean = float(y_train.mean())
y_std = float(y_train.std()) if y_train.std() > 0 else 1.0
y_train_s = (y_train - y_mean) / y_std

def build_model(input_dim):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)
    r = layers.Dense(128, activation='relu')(x)
    r = layers.Dense(128, activation=None)(r)
    x = layers.Add()([x, r])
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(1, activation='linear')(x)
    return models.Model(inputs=inp, outputs=out)

for i in range(args.nn_count):
    tf.keras.utils.set_random_seed(args.random_state + i)
    model = build_model(X_train_s.shape[1])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
    es = callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
    model.fit(X_train_s, y_train_s, validation_split=0.1, epochs=200, batch_size=32, callbacks=[es], verbose=1)
    p = os.path.join(OUT_DIR, f'nn_model_{i}.h5')
    model.save(p)
    nn_paths.append(p)

joblib.dump({'model_paths': nn_paths, 'y_mean': y_mean, 'y_std': y_std}, os.path.join(OUT_DIR, 'nn_ensemble.meta.joblib'))

print("Training complete. Models saved to:", OUT_DIR)
