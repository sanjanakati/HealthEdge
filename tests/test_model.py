"""Smoke test: the demo examples must behave as documented. Run: pytest"""
import subprocess, sys, os, joblib
ROOT = os.path.dirname(os.path.dirname(__file__))

def test_demo_examples():
    subprocess.run([sys.executable, "train_model.py"], cwd=ROOT, check=True)
    model = joblib.load(os.path.join(ROOT, "healthedge_model.joblib"))
    assert model.predict([[76, 36.8]])[0] == 1      # normal
    assert model.predict([[135, 39.2]])[0] == -1    # anomaly
