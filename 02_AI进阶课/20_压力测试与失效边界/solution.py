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
# ### 1. 写出可手算的策略评价函数
import numpy as np
import pandas as pd

def evaluate_policy(policy, demand, failure):
    base, reserve = policy
    available = np.maximum(0, base - failure)
    activate = np.minimum(reserve, np.maximum(0, np.ceil(demand / 10) - available))
    shortage = np.maximum(0, demand - 10 * (available + activate))
    cost = base * 12 + reserve * 3 + activate * 8 + shortage * 15
    return (cost, shortage)
assert evaluate_policy((2, 0), np.array([0]), np.array([0]))[0][0] == 24
assert evaluate_policy((2, 0), np.array([30]), np.array([0]))[1][0] == 10
policies = {'lean': (10, 0), 'reserve': (10, 3), 'large': (13, 0)}

# %%
# ### 2. 留出多类机制变化
def scenario(kind, seed):
    rng = np.random.default_rng(seed)
    d = rng.poisson(85, (100, 8)).astype(float)
    f = rng.binomial(10, 0.03, (100, 8))
    if kind == 'higher':
        d += 25
    if kind == 'heavy_tail':
        d += rng.binomial(1, 0.05, (100, 8)) * 80
    if kind == 'correlated_failure':
        f += rng.binomial(1, 0.15, (100, 1)) * 4
    return (d, f)
records = []
paired = {}
for i, kind in enumerate(['normal', 'higher', 'heavy_tail', 'correlated_failure']):
    d, f = scenario(kind, 2000 + i)
    for name, policy in policies.items():
        c, s = evaluate_policy(policy, d, f)
        daily = c.sum(axis=1)
        paired[kind, name] = daily
        records.append({'scenario': kind, 'policy': name, 'mean_cost': float(daily.mean()), 'p95_daily_cost': float(np.quantile(daily, 0.95)), 'all_periods_pass_rate': float(np.mean((s == 0).all(axis=1))), 'period_pass_rate': float(np.mean(s == 0))})
results = pd.DataFrame(records)

# %%
# ### 3. 成对差值区间与情景遗憾
rng = np.random.default_rng(20)
differences = paired['normal', 'reserve'] - paired['normal', 'lean']
boot = np.array([rng.choice(differences, len(differences), replace=True).mean() for _ in range(500)])
interval = np.quantile(boot, [0.025, 0.975])
results['scenario_regret'] = results.mean_cost - results.groupby('scenario').mean_cost.transform('min')
results.to_csv(OUT_DIR / 'stress_matrix.csv', index=False)
pivot = results.pivot(index='scenario', columns='policy', values='mean_cost')
pivot = pivot.rename(index={'normal': '常态', 'higher': '需求抬升', 'heavy_tail': '厚尾需求', 'correlated_failure': '相关故障'}, columns={'lean': '精简配置', 'reserve': '预留备用', 'large': '常备增员'})
pivot.plot.bar()
plt.xlabel('压力情景')
plt.ylabel('平均每日成本（元，8时段合计）')
plt.xticks(rotation=0)
plt.legend(title='策略')
plt.tight_layout()
plt.savefig(OUT_DIR / 'stress.png')
plt.close()

# %%
# ### 4. 说明风险偏好与边界
metrics = {'cost_unit': '元/活动日（8时段合计）', 'normal_reserve_minus_lean_mean': float(differences.mean()), 'bootstrap_95_interval': interval.tolist(), 'stress_results': results.to_dict('records')}
recommendation = '请依据 stress_matrix.csv 同时比较平均成本、95%分位日成本与整日全通过率。若所有时段都必须有容量，优先检查 all_periods_pass_rate；表中出现缺口的方案不能因平均得分高就称为满足硬约束。'

# %%
# ### 保存可核验的证据
import json
(OUT_DIR / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
(OUT_DIR / '决策建议.md').write_text('# 本次计算的决策建议\n\n' + recommendation + '\n\n## 实际指标\n```json\n' + json.dumps(metrics, ensure_ascii=False, indent=2) + '\n```\n\n这些结论只适用于本例显式假设；目标权重和风险偏好由人决定。', encoding='utf-8')
print(metrics)