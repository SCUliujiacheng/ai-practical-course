from pathlib import Path
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
CASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
COURSE_ROOT = CASE_DIR.parents[1]
DATA_DIR = CASE_DIR / "data"
OUT_DIR = CASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                     "figure.figsize": (8, 4.5), "figure.dpi": 110})


# %%
# ## 1. 固定测试与比较问题
import json
import time
import numpy as np
started = time.perf_counter()
def finish(metrics):
    metrics["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
def barplot(labels, values, title, filename="comparison.png"):
    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename)
    plt.close(fig)

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
X, y = make_moons(n_samples=1200, noise=0.25, random_state=27)
tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=27)
seeds = list(range(8))
base_errors = []
new_errors = []
timings = []

# %%
# ## 2. 同一训练子样本上的配对比较
for seed in seeds:
    rng = np.random.default_rng(seed)
    subset = rng.choice(tr, size=int(0.8 * len(tr)), replace=False)
    base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    improved = make_pipeline(PolynomialFeatures(3, include_bias=False), StandardScaler(), LogisticRegression(C=1, max_iter=500))
    t0 = time.perf_counter()
    base.fit(X[subset], y[subset])
    base_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    improved.fit(X[subset], y[subset])
    new_time = time.perf_counter() - t0
    base_errors.append((base.predict(X[te]) != y[te]).astype(float))
    new_errors.append((improved.predict(X[te]) != y[te]).astype(float))
    timings.append((base_time, new_time))
base_errors = np.array(base_errors)
new_errors = np.array(new_errors)
per_seed_gain = base_errors.mean(1) - new_errors.mean(1)

# %%
# ## 3. 对测试样本做配对 bootstrap
per_case_gain = (base_errors - new_errors).mean(axis=0)

def paired_bootstrap(values, seed=27, repeats=1500):
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(repeats, len(values)), replace=True).mean(1)
    return (np.quantile(samples, [0.025, 0.975]), samples)
interval, bootstrap = paired_bootstrap(per_case_gain)
assert interval[0] <= interval[1]

# %%
# ## 4. 报告收益、变异与成本
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(seeds, per_seed_gain)
axes[0].axhline(0, color='black')
axes[0].set_title('各训练子样本的准确率提升')
axes[1].hist(bootstrap, bins=30)
axes[1].axvline(interval[0], color='red')
axes[1].axvline(interval[1], color='red')
axes[1].set_title('配对 bootstrap 均值分布')
fig.tight_layout()
fig.savefig(OUT_DIR / 'uncertainty.png')
plt.close(fig)
np.savez(OUT_DIR / 'paired_errors.npz', baseline=base_errors, improved=new_errors, test_indices=te)
finish({'baseline_accuracy': float(1 - base_errors.mean()), 'improved_accuracy': float(1 - new_errors.mean()), 'mean_accuracy_gain': float(per_case_gain.mean()), 'paired_95_percent_interval': interval.tolist(), 'per_seed_gain': per_seed_gain.tolist(), 'mean_training_seconds': np.mean(timings, axis=0).tolist(), 'interval_scope': 'fixed trained models; iid test examples; not full training uncertainty'})