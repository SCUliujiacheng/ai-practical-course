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
# ### 1. 从需求分布开始
import numpy as np
import pandas as pd
rng = np.random.default_rng(17)

def sample_demand(rng, n):
    return np.maximum(0, np.rint(rng.normal(95, 18, n) + rng.binomial(1, 0.15, n) * 45)).astype(int)
train = sample_demand(rng, 500)
validation = sample_demand(rng, 250)
test = sample_demand(rng, 1000)
pd.DataFrame({'demand': train}).to_csv(DATA_DIR / 'historical_demand.csv', index=False)
underage_cost = 12.0
overage_cost = 4.0

# %%
# ### 2. 定义行动损失
def cost(quantity, demand, underage=underage_cost, overage=overage_cost):
    demand = np.asarray(demand)
    return underage * np.maximum(demand - quantity, 0) + overage * np.maximum(quantity - demand, 0)
assert cost(10, 8) == 8 and cost(10, 12) == 24
critical_ratio = underage_cost / (underage_cost + overage_cost)
quantile_quantity = int(np.quantile(train, critical_ratio, method='higher'))
mean_quantity = int(np.rint(train.mean()))

# %%
# ### 3. 验证选量与最终评价分开
candidates = np.arange(60, 161)
validation_losses = np.array([cost(q, validation).mean() for q in candidates])
selected = int(candidates[validation_losses.argmin()])
policies = {'mean': mean_quantity, 'critical_quantile': quantile_quantity, 'validation_search': selected}
rows = []
for name, q in policies.items():
    rows.append({'policy': name, 'quantity': q, 'test_cost': cost(q, test).mean(), 'stockout_rate': float(np.mean(test > q)), 'leftovers': np.maximum(q - test, 0).mean()})
results = pd.DataFrame(rows)
results.to_csv(OUT_DIR / 'policies.csv', index=False)
plt.plot(candidates, validation_losses)
plt.axvline(selected, color='orange', label='验证集选量')
plt.xlabel('备货份数')
plt.ylabel('验证平均损失（元）')
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / 'cost_curve.png')
plt.close()

# %%
# ### 4. 需求分布变化会改变建议
metrics = {'critical_ratio': critical_ratio, 'selected_quantity': selected, 'test_results': results.to_dict('records'), 'shifted_demand_cost': float(cost(selected, test + 20).mean())}
recommendation = f'验证数据推荐备货 {selected} 份，独立测试平均损失 {cost(selected, test).mean():.2f} 元/日。需求整体增加20份时损失变为 {cost(selected, test + 20).mean():.2f} 元/日；发现需求变化后应重新估计分布。'

# %%
# ### 保存可核验的证据
import json
(OUT_DIR / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
(OUT_DIR / '决策建议.md').write_text('# 本次计算的决策建议\n\n' + recommendation + '\n\n## 实际指标\n```json\n' + json.dumps(metrics, ensure_ascii=False, indent=2) + '\n```\n\n这些结论只适用于本例显式假设；目标权重和风险偏好由人决定。', encoding='utf-8')
print(metrics)