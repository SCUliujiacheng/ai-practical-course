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
# ## 1. 固定少样本训练与测试
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

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
torch.set_num_threads(2)
torch.manual_seed(19)
d = load_digits()
X = (d.images / 16).astype('float32')
y = d.target
rng = np.random.default_rng(19)
tr = []
te = []
for label in range(10):
    idx = rng.permutation(np.flatnonzero(y == label))
    tr.extend(idx[:15])
    te.extend(idx[15:25])
tr = np.array(tr)
te = np.array(te)
assert set(tr).isdisjoint(te)

# %%
# ## 2. 从本地加载权重并冻结
weights_path = COURSE_ROOT / 'assets/models/resnet18.pth'
if not weights_path.is_file():
    raise FileNotFoundError(f'缺少离线权重：{weights_path}')
backbone = resnet18(weights=None)
backbone.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
backbone.fc = nn.Identity()
for p in backbone.parameters():
    p.requires_grad = False
backbone.eval()
assert all((not p.requires_grad for p in backbone.parameters()))

# %%
# ## 3. 统一预处理后提取特征
def extract_features(images):
    pieces = []
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    with torch.no_grad():
        for start in range(0, len(images), 25):
            batch = torch.tensor(images[start:start + 25])[:, None].repeat(1, 3, 1, 1)
            batch = F.interpolate(batch, size=(112, 112), mode='bilinear', align_corners=False)
            pieces.append(backbone((batch - mean) / std).numpy())
    return np.concatenate(pieces)
train_features = extract_features(X[tr])
test_features = extract_features(X[te])
assert train_features.shape == (150, 512)
np.savez(OUT_DIR / 'features.npz', train=train_features, test=test_features)

# %%
# ## 4. 只训练新分类头，并解释迁移结果
raw = LogisticRegression(max_iter=500).fit(X[tr].reshape(len(tr), -1), y[tr])
head = LogisticRegression(max_iter=600, C=0.5).fit(train_features, y[tr])
raw_pred = raw.predict(X[te].reshape(len(te), -1))
transfer_pred = head.predict(test_features)
scores = [float(accuracy_score(y[te], raw_pred)), float(accuracy_score(y[te], transfer_pred))]
barplot(['原像素', '冻结 ResNet18'], scores, '少样本迁移：测试准确率')
finish({'raw_pixel_accuracy': scores[0], 'frozen_resnet_accuracy': scores[1], 'train_samples': len(tr), 'test_samples': len(te), 'backbone_trainable_parameters': 0, 'input_size': 112, 'domain_warning': 'ImageNet to digits; teaching workflow only'})