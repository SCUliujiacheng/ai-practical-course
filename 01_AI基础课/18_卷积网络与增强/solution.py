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
# ## 1. 保留空间结构与验证集
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

import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
torch.set_num_threads(2)
torch.manual_seed(18)
d = load_digits()
X = (d.images / 16).astype('float32')[:, None, :, :]
y = d.target
rest, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=18, stratify=y)
tr, va = train_test_split(rest, test_size=0.25, random_state=18, stratify=y[rest])
assert set(tr).isdisjoint(va) and set(rest).isdisjoint(te)
print(X[tr].shape)

# %%
# ## 2. 写出卷积、池化和分类头
def build_cnn():
    return nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(16 * 4 * 4, 10))

def build_mlp():
    return nn.Sequential(nn.Flatten(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))

def shift_right(batch):
    result = torch.zeros_like(batch)
    result[:, :, :, 1:] = batch[:, :, :, :-1]
    return result
assert build_cnn()(torch.zeros(3, 1, 8, 8)).shape == (3, 10)

# %%
# ## 3. 只对训练图像增强
def train_network(network, augment=False, epochs=10):
    torch.manual_seed(18)
    loader = DataLoader(TensorDataset(torch.tensor(X[tr]), torch.tensor(y[tr], dtype=torch.long)), batch_size=64, shuffle=True)
    opt = torch.optim.Adam(network.parameters(), lr=0.006)
    criterion = nn.CrossEntropyLoss()
    best = float('inf')
    state = None
    trace = []
    for epoch in range(epochs):
        network.train()
        for bx, by in loader:
            if augment:
                mask = torch.rand(len(bx)) < 0.2
                bx = bx.clone()
                bx[mask] = shift_right(bx[mask])
            opt.zero_grad()
            loss = criterion(network(bx), by)
            loss.backward()
            opt.step()
        network.eval()
        with torch.no_grad():
            val = float(criterion(network(torch.tensor(X[va])), torch.tensor(y[va], dtype=torch.long)))
        trace.append(val)
        if val < best:
            best = val
            state = copy.deepcopy(network.state_dict())
    network.load_state_dict(state)
    return (network, trace)
mlp, mlp_history = train_network(build_mlp())
cnn, cnn_history = train_network(build_cnn(), augment=True)

# %%
# ## 4. 最后比较独立测试结果
with torch.no_grad():
    p_mlp = mlp(torch.tensor(X[te])).argmax(1).numpy()
    p_cnn = cnn(torch.tensor(X[te])).argmax(1).numpy()
fig, ax = plt.subplots()
ax.plot(mlp_history, label='MLP')
ax.plot(cnn_history, label='CNN + 平移')
ax.set_title('验证损失')
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / 'validation.png')
plt.close(fig)
finish({'mlp_accuracy': float(accuracy_score(y[te], p_mlp)), 'cnn_accuracy': float(accuracy_score(y[te], p_cnn)), 'mlp_parameters': sum((p.numel() for p in mlp.parameters())), 'cnn_parameters': sum((p.numel() for p in cnn.parameters())), 'selected_by': 'validation loss'})