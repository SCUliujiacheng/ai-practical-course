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
# ### 1. 列出状态、动作与转移
import numpy as np
import pandas as pd
from functools import lru_cache
CAPACITY = 8
support = np.array([0, 2, 4])
probability = np.array([0.15, 0.45, 0.4])

def transition(stock, order, demand):
    assert 0 <= order <= CAPACITY - stock
    available = stock + order
    left = max(available - demand, 0)
    short = max(demand - available, 0)
    return (left, 2 * order + 9 * short + 0.5 * left)
assert transition(2, 2, 3) == (1, 4.5)

# %%
# ### 2. 倒推未来价值
@lru_cache(None)
def optimal(days, stock):
    if days == 0:
        return (0.0, 0)
    candidates = []
    for order in range(CAPACITY - stock + 1):
        value = 0.0
        for demand, p in zip(support, probability):
            nxt, now = transition(stock, order, int(demand))
            value += p * (now + optimal(days - 1, nxt)[0])
        candidates.append((value, order))
    return min(candidates)
policy_table = pd.DataFrame([{'remaining_days': days, 'stock': stock, 'order': optimal(days, stock)[1], 'value': optimal(days, stock)[0]} for days in [1, 2, 3] for stock in range(CAPACITY + 1)])
policy_table.to_csv(OUT_DIR / 'policy.csv', index=False)

# %%
# ### 3. 执行反馈而非照抄整条计划
def rollout(demands, mode):
    stock = 0
    total = 0.0
    trace = []
    for day, demand in enumerate(demands):
        remaining = len(demands) - day
        if mode == 'full':
            horizon = remaining
        elif mode == 'rolling2':
            horizon = min(2, remaining)
        else:
            horizon = 1
        order = optimal(horizon, stock)[1]
        nxt, cost = transition(stock, order, int(demand))
        trace.append({'day': day + 1, 'stock': stock, 'order': order, 'observed_demand': int(demand), 'next_stock': nxt, 'cost': cost})
        stock = nxt
        total += cost
    return (total, trace)
rng = np.random.default_rng(26)
paths = rng.choice(support, (600, 3), p=probability)
records = []
for mode in ['full', 'rolling2', 'myopic']:
    costs = np.array([rollout(d, mode)[0] for d in paths])
    records.append({'mode': mode, 'mean': costs.mean(), 'std': costs.std()})
comparison = pd.DataFrame(records)
pd.DataFrame(rollout([4, 0, 4], 'rolling2')[1]).to_csv(OUT_DIR / 'execution_trace.csv', index=False)
comparison.plot.bar(x='mode', y='mean', legend=False)
plt.ylabel('三天总成本（元）')
plt.tight_layout()
plt.savefig(OUT_DIR / 'horizon.png')
plt.close()

# %%
# ### 4. 检查理论值与模拟
full_costs = np.array([rollout(d, 'full')[0] for d in paths])
se = full_costs.std(ddof=1) / np.sqrt(len(paths))
assert abs(full_costs.mean() - optimal(3, 0)[0]) < 5 * se + 0.1
metrics = {'exact_optimal_expected_cost': optimal(3, 0)[0], 'simulation': comparison.to_dict('records'), 'heavy_demand_total': rollout([4, 4, 4], 'rolling2')[0]}
recommendation = f'已知小模型的最优期望成本为 {optimal(3, 0)[0]:.2f} 元。滚动策略应在每天观察需求后重新求解；若需求机制变化，需要更新概率模型，不能仅增加求解次数。'

# %%
# ### 5. 什么条件下看得更远才有价值？
@lru_cache(None)
def optimal_with_fee(days, stock, fee):
    if days == 0:
        return (0.0, 0)
    candidates = []
    for order in range(CAPACITY - stock + 1):
        value = 0.0
        for demand, p in zip(support, probability):
            nxt, now = transition(stock, order, int(demand))
            now += fee if order > 0 else 0
            value += p * (now + optimal_with_fee(days - 1, nxt, fee)[0])
        candidates.append((value, order))
    return min(candidates)

def rollout_with_fee(demands, mode, fee=8.0):
    stock = 0
    total = 0.0
    for day, demand in enumerate(demands):
        remaining = len(demands) - day
        horizon = remaining if mode == 'full' else min(2, remaining) if mode == 'rolling2' else 1
        order = optimal_with_fee(horizon, stock, fee)[1]
        stock, now = transition(stock, order, int(demand))
        total += now + (fee if order > 0 else 0)
    return total
fee_comparison = []
for mode in ['full', 'rolling2', 'myopic']:
    costs = np.array([rollout_with_fee(d, mode) for d in paths])
    fee_comparison.append({'mode': mode, 'mean': float(costs.mean()), 'std': float(costs.std())})
assert optimal_with_fee(3, 0, 0) == optimal(3, 0)
assert optimal_with_fee(3, 0, 8)[1] == 8 and optimal_with_fee(1, 0, 8)[1] == 4
pd.DataFrame(fee_comparison).to_csv(OUT_DIR / 'fixed_fee_comparison.csv', index=False)
pd.DataFrame(fee_comparison).plot.bar(x='mode', y='mean', legend=False, rot=0)
plt.xlabel('规划视野')
plt.ylabel('三天总成本（元）')
plt.title('加入每次8元配送费后的独立模拟')
plt.tight_layout()
plt.savefig(OUT_DIR / 'fixed_fee.png')
plt.close()
metrics['fixed_order_fee'] = 8
metrics['with_fee_exact_optimal_expected_cost'] = optimal_with_fee(3, 0, 8)[0]
metrics['with_fee_simulation'] = fee_comparison
recommendation += ' 基础设置中三种视野得到相同策略；加入每次8元配送费后再比较 fixed_fee_comparison.csv，理解跨日耦合为何改变规划价值。'

# %%
# ### 保存可核验的证据
import json
(OUT_DIR / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
(OUT_DIR / '决策建议.md').write_text('# 本次计算的决策建议\n\n' + recommendation + '\n\n## 实际指标\n```json\n' + json.dumps(metrics, ensure_ascii=False, indent=2) + '\n```\n\n这些结论只适用于本例显式假设；目标权重和风险偏好由人决定。', encoding='utf-8')
print(metrics)