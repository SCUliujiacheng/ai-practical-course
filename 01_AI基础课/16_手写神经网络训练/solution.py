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
# ## 1. 把数据变成形状明确的 Tensor
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
torch.set_num_threads(2)
torch.manual_seed(16)
rng = np.random.default_rng(16)
x = rng.uniform(-2, 2, (240, 1)).astype('float32')
y = (0.8 * x * x + 0.5 * x + rng.normal(0, 0.12, x.shape)).astype('float32')
x_train, y_train = (torch.from_numpy(x[:160]), torch.from_numpy(y[:160]))
x_test, y_test = (torch.from_numpy(x[160:]), torch.from_numpy(y[160:]))
assert x_train.shape == y_train.shape == (160, 1)
np.savez(DATA_DIR / 'sensor.npz', x=x, y=y)
print('训练形状：', x_train.shape)

# %%
# ## 2. 从一个导数到网络参数
w = torch.tensor(1.0, requires_grad=True)
((w - 3) ** 2).backward()
assert w.grad.item() == -4.0
model = nn.Sequential(nn.Linear(1, 24), nn.ReLU(), nn.Linear(24, 1))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
print('可训练参数：', sum((p.numel() for p in model.parameters())))

# %%
# ## 3. 写出每一次更新
losses = []
for epoch in range(220):
    model.train()
    prediction = model(x_train)
    loss = criterion(prediction, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach()))
assert np.isfinite(losses).all() and losses[-1] < losses[0]

# %%
# ## 4. 在未参与训练的数据上检查
model.eval()
with torch.no_grad():
    predicted = model(x_test).numpy().ravel()
baseline = np.full(len(predicted), float(y_train.mean()))
mse = float(np.mean((predicted - y[160:, 0]) ** 2))
baseline_mse = float(np.mean((baseline - y[160:, 0]) ** 2))
torch.save(model.state_dict(), OUT_DIR / 'network.pt')
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(losses)
axes[0].set_title('训练损失')
axes[0].set_xlabel('更新次数')
order = np.argsort(x[160:, 0])
axes[1].scatter(x[160:, 0], y[160:, 0], s=12, label='观测')
axes[1].plot(x[160:, 0][order], predicted[order], color='red', label='预测')
axes[1].legend()
fig.tight_layout()
fig.savefig(OUT_DIR / 'training.png')
plt.close(fig)
finish({'test_mse': mse, 'mean_baseline_mse': baseline_mse, 'train_first_loss': losses[0], 'train_last_loss': losses[-1]})