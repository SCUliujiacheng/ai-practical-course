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
# ## 1. 固定模型与三个监测窗口
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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import ks_2samp
rng = np.random.default_rng(29)
reference = rng.normal(size=(1000, 2))
reference_y = (reference[:, 0] + 0.6 * reference[:, 1] + rng.normal(0, 0.4, 1000) > 0).astype(int)
model = LogisticRegression().fit(reference[:700], reference_y[:700])
windows = {}
for name, shift, flip in [('正常', 0, False), ('输入偏移', 1.5, False), ('概念改变', 0, True)]:
    xx = rng.normal(size=(400, 2))
    xx[:, 0] += shift
    yy = (xx[:, 0] + 0.6 * xx[:, 1] + rng.normal(0, 0.4, 400) > 0).astype(int)
    if flip:
        yy = 1 - yy
    windows[name] = (xx, yy)
baseline_accuracy = float(accuracy_score(reference_y[700:], model.predict(reference[700:])))

# %%
# ## 2. 输入质量先于预测
def quality_mask(X):
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError('要求两列特征')
    return np.isfinite(X).all(axis=1)
broken = windows['正常'][0].copy()
broken[0, 0] = np.nan
broken[1, 1] = np.inf
valid = quality_mask(broken)
assert valid.sum() == len(broken) - 2

# %%
# ## 3. 标签前后使用不同指标
def monitor(X, labels=None):
    mask = quality_mask(X)
    clean = X[mask]
    if len(clean) == 0:
        raise ValueError('没有可用记录')
    ks = [float(ks_2samp(reference[:, j], clean[:, j]).statistic) for j in range(2)]
    accuracy = None if labels is None else float(accuracy_score(np.asarray(labels)[mask], model.predict(clean)))
    return {'valid_rows': int(mask.sum()), 'invalid_rows': int((~mask).sum()), 'ks_statistics': ks, 'input_drift': max(ks) > 0.2, 'accuracy': accuracy, 'performance_alarm': accuracy is not None and accuracy < baseline_accuracy - 0.1}
unlabeled = {n: monitor(x) for n, (x, y) in windows.items()}
labeled = {n: monitor(x, y) for n, (x, y) in windows.items()}
assert all((r['accuracy'] is None for r in unlabeled.values()))

# %%
# ## 4. 从报警到行动
report = {'reference_accuracy': baseline_accuracy, 'before_labels': unlabeled, 'after_labels': labeled, 'quality_example': monitor(broken)}
(OUT_DIR / 'monitoring_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
names = list(windows)
axes[0].bar(names, [max(labeled[n]['ks_statistics']) for n in names])
axes[0].axhline(0.2, color='red')
axes[0].set_title('输入漂移 KS')
axes[1].bar(names, [labeled[n]['accuracy'] for n in names])
axes[1].axhline(baseline_accuracy, color='red')
axes[1].set_title('标签到达后的准确率')
fig.tight_layout()
fig.savefig(OUT_DIR / 'monitoring.png')
plt.close(fig)
finish({'reference_accuracy': baseline_accuracy, 'window_results': labeled, 'invalid_rows_detected': int((~valid).sum()), 'thresholds': 'teaching assumptions; calibrate for real deployment'})