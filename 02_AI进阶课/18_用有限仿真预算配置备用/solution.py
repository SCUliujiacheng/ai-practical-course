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
# ### 1. 定义可行动的策略参数
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(2)

def fleet_cost(reserve, threshold, seeds):
    losses = []
    late_rates = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        demand = rng.poisson(95, 30)
        failures = rng.binomial(10, 0.08, 30)
        base_capacity = (10 - failures) * 10
        deploy = np.where(demand - base_capacity > threshold, reserve, 0)
        late = np.maximum(0, demand - base_capacity - 10 * deploy)
        losses.append(float(np.mean(8 * late + reserve * 4 + deploy * 2)))
        late_rates.append(float(late.sum() / demand.sum()))
    return (float(np.mean(losses)), float(np.mean(late_rates)))
train_seeds = list(range(180, 190))
test_seeds = list(range(500, 550))
budget = 10
assert fleet_cost(2, 8, [1]) == fleet_cost(2, 8, [1])

# %%
# ### 2. 建立同预算随机搜索
rng = np.random.default_rng(18)
random_trials = []
for i in range(budget):
    parameters = {'reserve': int(rng.integers(0, 7)), 'threshold': int(rng.integers(0, 26))}
    value, _ = fleet_cost(**parameters, seeds=train_seeds)
    random_trials.append(dict(trial=i, **parameters, cost=value))
random_best = min(random_trials, key=lambda row: row['cost'])

# %%
# ### 3. 用Ax逐步选试验
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models
strategy = GenerationStrategy(steps=[GenerationStep(model=Models.SOBOL, num_trials=4, model_kwargs={'seed': 18}), GenerationStep(model=Models.BOTORCH_MODULAR, num_trials=-1, model_gen_kwargs={'model_gen_options': {'optimizer_kwargs': {'num_restarts': 2, 'raw_samples': 32}}})])
client = AxClient(generation_strategy=strategy, random_seed=18, verbose_logging=False)
client.create_experiment(name='fleet_reserve', parameters=[{'name': 'reserve', 'type': 'range', 'bounds': [0, 6], 'value_type': 'int'}, {'name': 'threshold', 'type': 'range', 'bounds': [0, 25], 'value_type': 'int'}], objectives={'cost': ObjectiveProperties(minimize=True)})
ax_trials = []
for i in range(budget):
    parameters, trial_index = client.get_next_trial()
    value, _ = fleet_cost(**parameters, seeds=train_seeds)
    client.complete_trial(trial_index=trial_index, raw_data={'cost': (value, 0.0)})
    ax_trials.append(dict(trial=i, **parameters, cost=value))
ax_best = min(ax_trials, key=lambda row: row['cost'])

# %%
# ### 4. 只在选型之后独立复评
evaluation = []
for name, row in [('random', random_best), ('ax', ax_best), ('no_reserve', {'reserve': 0, 'threshold': 0})]:
    value, late = fleet_cost(row['reserve'], row['threshold'], test_seeds)
    evaluation.append({'method': name, 'reserve': int(row['reserve']), 'threshold': int(row['threshold']), 'test_cost': value, 'late_rate': late})
pd.DataFrame(evaluation).to_csv(OUT_DIR / 'independent_evaluation.csv', index=False)
for name, trials in [('random', random_trials), ('Ax', ax_trials)]:
    plt.plot(range(1, budget + 1), np.minimum.accumulate([row['cost'] for row in trials]), marker='o', label=name)
plt.xlabel('已用仿真目标调用次数')
plt.ylabel('当前最好校准损失')
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / 'search.png')
plt.close()
metrics = {'calls_per_method': budget, 'initial_sobol_trials': 4, 'bayesian_trials': 6, 'independent_results': evaluation}
recommendation = '两个搜索方法各使用10次目标调用，候选通过独立情景复评。请按 independent_evaluation.csv 比较损失和延误率；本次小预算实验不支持宣布某搜索方法普遍更优。'

# %%
# ### 保存可核验的证据
import json
(OUT_DIR / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
(OUT_DIR / '决策建议.md').write_text('# 本次计算的决策建议\n\n' + recommendation + '\n\n## 实际指标\n```json\n' + json.dumps(metrics, ensure_ascii=False, indent=2) + '\n```\n\n这些结论只适用于本例显式假设；目标权重和风险偏好由人决定。', encoding='utf-8')
print(metrics)