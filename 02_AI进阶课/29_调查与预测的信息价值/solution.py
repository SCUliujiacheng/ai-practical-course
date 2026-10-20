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
# ### 1. 定义状态、动作与损失
import numpy as np
import pandas as pd
prior = np.array([0.65, 0.35])
loss = np.array([[0.0, 1200.0], [350.0, 350.0]])

def choose(probability):
    expected = loss @ probability
    return (int(expected.argmin()), float(expected.min()))
action, baseline_loss = choose(prior)
assert np.isclose(prior.sum(), 1)

# %%
# ### 2. 完美信息给出上界
perfect_loss = float(prior @ loss.min(axis=0))
evpi = baseline_loss - perfect_loss
assert evpi >= 0

# %%
# ### 3. 噪声信号与贝叶斯更新
likelihood = np.array([[0.8, 0.25], [0.2, 0.75]])
assert np.allclose(likelihood.sum(axis=0), 1)
records = []
after_information_loss = 0.0
for signal in range(2):
    joint = likelihood[signal] * prior
    signal_probability = float(joint.sum())
    posterior = joint / signal_probability
    chosen, conditional_loss = choose(posterior)
    after_information_loss += signal_probability * conditional_loss
    records.append({'signal': signal, 'signal_probability': signal_probability, 'posterior_rain': float(posterior[1]), 'action': chosen, 'conditional_loss': conditional_loss})
evsi = baseline_loss - after_information_loss
assert -1e-08 <= evsi <= evpi + 1e-08
pd.DataFrame(records).to_csv(OUT_DIR / 'signal_decisions.csv', index=False)

# %%
# ### 4. 信息需要改变行动才可能有价值
prices = np.arange(0, 301, 10)
plt.plot(prices, evsi - prices)
plt.axhline(0, color='gray')
plt.xlabel('调查价格（元）')
plt.ylabel('购买信息的期望净价值（元）')
plt.tight_layout()
plt.savefig(OUT_DIR / 'information_value.png')
plt.close()
metrics = {'baseline_action': action, 'baseline_loss': baseline_loss, 'perfect_information_value': evpi, 'noisy_information_value': evsi, 'signal_policy': records}
recommendation = f'当前先验下直接行动的最小期望损失为{baseline_loss:.2f}元；该噪声信号价值为{evsi:.2f}元，因此在其他成本不变时购买价格应低于该值。完美信息价值{evpi:.2f}元只是上界。'

# %%
# ### 保存可核验的证据
import json
(OUT_DIR / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
(OUT_DIR / '决策建议.md').write_text('# 本次计算的决策建议\n\n' + recommendation + '\n\n## 实际指标\n```json\n' + json.dumps(metrics, ensure_ascii=False, indent=2) + '\n```\n\n这些结论只适用于本例显式假设；目标权重和风险偏好由人决定。', encoding='utf-8')
print(metrics)