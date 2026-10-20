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
# ### 1. 反馈机制
import numpy as np
import pandas as pd
rates = np.array([0.4, 0.48, 0.58])
horizon = 600

def run_bandit(method, seed, steps=horizon):
    rng = np.random.default_rng(seed)
    counts = np.zeros(3, int)
    success = np.zeros(3)
    rewards = []
    regret = []
    for t in range(steps):
        average = np.divide(success, counts, out=np.zeros(3), where=counts > 0)
        if t < 3:
            action = t
        elif method == 'random':
            action = int(rng.integers(3))
        elif method == 'epsilon' and rng.random() < 0.1:
            action = int(rng.integers(3))
        elif method == 'ucb':
            action = int(np.argmax(average + np.sqrt(2 * np.log(t + 1) / counts)))
        else:
            action = int(np.argmax(average))
        reward = int(rng.random() < rates[action])
        counts[action] += 1
        success[action] += reward
        rewards.append(reward)
        regret.append(float(rates.max() - rates[action]))
    return (np.cumsum(rewards), np.cumsum(regret), counts)

# %%
# ### 2. 单次轨迹看直觉
methods = ['random', 'greedy', 'epsilon', 'ucb']
for method in methods:
    reward, regret, counts = run_bandit(method, 28)
    assert counts.sum() == horizon and np.all(np.diff(regret) >= -1e-10)
    plt.plot(regret, label=method)
plt.xlabel('决策轮次')
plt.ylabel('累计伪遗憾')
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / 'one_run.png')
plt.close()

# %%
# ### 3. 多次独立重复
rows = []
for method in methods:
    for seed in range(100):
        reward, regret, counts = run_bandit(method, seed + 2800)
        rows.append({'method': method, 'seed': seed, 'successes': int(reward[-1]), 'regret': float(regret[-1]), 'best_arm_selections': int(counts[2])})
results = pd.DataFrame(rows)
summary = results.groupby('method').agg(successes=('successes', 'mean'), regret=('regret', 'mean'), regret_std=('regret', 'std'))
results.to_csv(OUT_DIR / 'trials.csv', index=False)
summary.regret.plot.bar()
plt.ylabel('600轮平均伪遗憾')
plt.tight_layout()
plt.savefig(OUT_DIR / 'repeat.png')
plt.close()

# %%
# ### 4. 区分短期目标与长期学习
metrics = {'true_rates': rates.tolist(), 'horizon': horizon, 'runs_per_method': 100, 'summary': summary.reset_index().to_dict('records')}
recommendation = '根据 trials.csv 选择与时间预算和探索代价相符的策略。本例动作不改变后续状态，使用老虎机即可；若行动影响未来库存或用户状态，需重新建立序贯模型。'

# %%
# ### 保存可核验的证据
import json
(OUT_DIR / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
(OUT_DIR / '决策建议.md').write_text('# 本次计算的决策建议\n\n' + recommendation + '\n\n## 实际指标\n```json\n' + json.dumps(metrics, ensure_ascii=False, indent=2) + '\n```\n\n这些结论只适用于本例显式假设；目标权重和风险偏好由人决定。', encoding='utf-8')
print(metrics)