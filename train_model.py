"""HealthEdge - train the Isolation Forest on synthetic NORMAL readings.

Run:  python train_model.py
Creates: data/healthedge_dataset.csv  and  healthedge_model.joblib
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

os.makedirs("data", exist_ok=True)

# 1. Synthetic normal readings (no real patient data is used)
rng = np.random.default_rng(42)
pulse = np.clip(rng.normal(75, 8, 2000), 55, 100)
temp = np.clip(rng.normal(36.8, 0.3, 2000), 36.1, 37.5)
df = pd.DataFrame({"pulse_bpm": pulse.round(1), "temp_c": temp.round(2)})
df.to_csv("data/healthedge_dataset.csv", index=False)

# 2. Train on normal data only (unsupervised)
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(df[["pulse_bpm", "temp_c"]].values)

# 3. Save the model for local (edge) inference
joblib.dump(model, "healthedge_model.joblib")
print("Dataset saved  -> data/healthedge_dataset.csv  (%d rows)" % len(df))
print("Model saved    -> healthedge_model.joblib")
