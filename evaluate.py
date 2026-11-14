"""HealthEdge evaluation: labelled synthetic test set -> metrics, latency, model size.
Run:  python evaluate.py
Produces: data/healthedge_dataset.csv, data/healthedge_test.csv,
          healthedge_model.joblib, results.json
Timing numbers depend on your machine.
"""
import json, time, os
os.makedirs("data", exist_ok=True)
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

SEED = 42
rng = np.random.default_rng(SEED)

def normal(n):
    pulse = np.clip(rng.normal(75, 8, n), 55, 100)
    temp = np.clip(rng.normal(36.8, 0.3, n), 36.1, 37.5)
    return pulse, temp

def anomalies(n):
    """Five abnormal patterns, n//5 each."""
    k = n // 5
    p, t = [], []
    # 1 tachycardia + fever
    p.append(rng.uniform(115, 160, k)); t.append(rng.uniform(38.3, 40.5, k))
    # 2 tachycardia only
    p.append(rng.uniform(120, 170, k)); t.append(rng.normal(36.8, 0.3, k))
    # 3 high fever only
    p.append(rng.normal(78, 8, k));     t.append(rng.uniform(38.5, 40.5, k))
    # 4 bradycardia
    p.append(rng.uniform(30, 48, k));   t.append(rng.normal(36.7, 0.3, k))
    # 5 hypothermia + low pulse
    p.append(rng.uniform(40, 55, k));   t.append(rng.uniform(33.5, 35.4, k))
    return np.concatenate(p), np.concatenate(t)

# ---- training data (normal only) ----
n_train = 2000
p, t = normal(n_train)
train = pd.DataFrame({"pulse_bpm": p.round(1), "temp_c": t.round(2)})
train.to_csv("data/healthedge_dataset.csv", index=False)

# ---- test data (labelled) ----
pn, tn = normal(500)
pa, ta = anomalies(125)
test = pd.DataFrame({
    "pulse_bpm": np.concatenate([pn, pa]).round(1),
    "temp_c": np.concatenate([tn, ta]).round(2),
    "label": [0]*500 + [1]*125})   # 1 = anomaly
test["pattern"] = ["normal"]*500 + sum([[n]*25 for n in
    ["tachy+fever","tachycardia","high fever","bradycardia","hypothermia"]], [])
test.to_csv("data/healthedge_test.csv", index=False)

def fit_eval(cont, n_est=100):
    m = IsolationForest(n_estimators=n_est, contamination=cont, random_state=SEED).fit(train[["pulse_bpm","temp_c"]].values)
    pred = (m.predict(test[["pulse_bpm","temp_c"]].values) == -1).astype(int)
    return m, pred

res = {}
model, pred = fit_eval(0.05)
y = test["label"].values
res["config"] = dict(n_estimators=100, contamination=0.05, random_state=SEED, n_train=n_train, n_test_normal=500, n_test_anom=125)
res["metrics"] = dict(accuracy=accuracy_score(y,pred), precision=precision_score(y,pred), recall=recall_score(y,pred), f1=f1_score(y,pred))
tn_, fp, fn, tp = confusion_matrix(y, pred).ravel()
res["confusion"] = dict(tn=int(tn_), fp=int(fp), fn=int(fn), tp=int(tp))
res["per_pattern_recall"] = {pt: float(pred[(test.pattern==pt).values].mean()) for pt in test.pattern.unique() if pt!="normal"}
res["false_alarm_rate"] = float(pred[y==0].mean())

# contamination sweep
sweep = []
for c in [0.01, 0.02, 0.05, 0.10, 0.15]:
    _, pr = fit_eval(c)
    sweep.append(dict(contamination=c, precision=precision_score(y,pr), recall=recall_score(y,pr), f1=f1_score(y,pr), far=float(pr[y==0].mean())))
res["sweep"] = sweep
# n_estimators sweep
nes = []
for ne in [10, 50, 100, 200]:
    _, pr = fit_eval(0.05, ne)
    nes.append(dict(n_estimators=ne, f1=f1_score(y,pr)))
res["n_est_sweep"] = nes

# rule-based baseline (fixed clinical-style thresholds)
rule = ((test.pulse_bpm<50)|(test.pulse_bpm>110)|(test.temp_c<35.5)|(test.temp_c>38.0)).astype(int).values
res["baseline_rule"] = dict(precision=precision_score(y,rule), recall=recall_score(y,rule), f1=f1_score(y,rule), far=float(rule[y==0].mean()))

# save + size
joblib.dump(model, "healthedge_model.joblib")
res["model_size_kb"] = os.path.getsize("healthedge_model.joblib")/1024

# training time
ts=[]
for _ in range(10):
    t0=time.perf_counter(); IsolationForest(n_estimators=100, contamination=0.05, random_state=SEED).fit(train[["pulse_bpm","temp_c"]].values); ts.append(time.perf_counter()-t0)
res["train_time_ms"] = float(np.mean(ts)*1000)

# single-reading latency (load from disk, then predict one reading, 1000 times)
t0=time.perf_counter(); loaded = joblib.load("healthedge_model.joblib"); res["load_time_ms"]=(time.perf_counter()-t0)*1000
x = np.array([[76,36.8]])
loaded.predict(x)  # warm-up
lat=[]
for _ in range(1000):
    t0=time.perf_counter(); loaded.predict(x); lat.append((time.perf_counter()-t0)*1000)
res["latency_ms"] = dict(mean=float(np.mean(lat)), p95=float(np.percentile(lat,95)), min=float(np.min(lat)))
res["throughput_per_s"] = 1000/np.mean(lat)

# the two demo examples
demo = {}
for name,(a,b) in {"normal":(76,36.8),"anomaly":(135,39.2)}.items():
    X=np.array([[a,b]]); demo[name]=dict(pulse=a,temp=b,pred=int(loaded.predict(X)[0]),score=float(loaded.decision_function(X)[0]))
res["demo"]=demo
res["lib_versions"]=dict(python=__import__("sys").version.split()[0], sklearn=__import__("sklearn").__version__, numpy=np.__version__, pandas=pd.__version__)
json.dump(res, open("results.json","w"), indent=2)
print(json.dumps(res, indent=2))

# ---- model-optimisation study: trees vs size / latency / F1 (Table 5 in the report) ----
opt = []
Xte = test[["pulse_bpm", "temp_c"]].values
for ne in [10, 25, 50, 100, 200]:
    m = IsolationForest(n_estimators=ne, contamination=0.05, random_state=SEED).fit(train[["pulse_bpm", "temp_c"]].values)
    f1v = f1_score(y, (m.predict(Xte) == -1).astype(int))
    joblib.dump(m, "_tmp.joblib"); size = os.path.getsize("_tmp.joblib") / 1024
    joblib.dump(m, "_tmpc.joblib", compress=3); sizec = os.path.getsize("_tmpc.joblib") / 1024
    l = joblib.load("_tmp.joblib"); x1 = np.array([[76, 36.8]]); l.predict(x1)
    lat = []
    for _ in range(300):
        t0 = time.perf_counter(); l.predict(x1); lat.append((time.perf_counter() - t0) * 1000)
    opt.append(dict(n_estimators=ne, f1=f1v, size_kb=size, size_compressed_kb=sizec, latency_ms=float(np.mean(lat))))
    print(f"trees={ne:>3}  F1={f1v:.3f}  size={size:.0f} KB  compressed={sizec:.0f} KB  latency={np.mean(lat):.2f} ms")
os.remove("_tmp.joblib"); os.remove("_tmpc.joblib")
res["opt"] = opt
json.dump(res, open("results.json", "w"), indent=2)
