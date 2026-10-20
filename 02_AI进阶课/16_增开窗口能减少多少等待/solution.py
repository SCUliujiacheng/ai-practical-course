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
# ### 1. 明确输入与单位
import numpy as np
import pandas as pd
import simpy
rng = np.random.default_rng(16)
calibration = pd.DataFrame({'interarrival': rng.exponential(2.0, 400), 'service': rng.gamma(3, 1.3, 400)})
holdout = pd.DataFrame({'interarrival': rng.exponential(2.0, 200), 'service': rng.gamma(3, 1.3, 200)})
calibration.to_csv(DATA_DIR / 'calibration.csv', index=False)
holdout.to_csv(DATA_DIR / 'holdout.csv', index=False)
arrival_mean = float(calibration.interarrival.mean())
service_mean = float(calibration.service.mean())
print('校准均值/留出均值', arrival_mean, service_mean, holdout.mean().to_dict())

# %%
# ### 2. 写出一个顾客的过程
def simulate(arrivals, services, capacity):
    assert capacity >= 1 and len(arrivals) == len(services)
    env = simpy.Environment()
    counter = simpy.Resource(env, capacity=capacity)
    waits = []

    def customer(arrival, service):
        yield env.timeout(float(arrival))
        with counter.request() as request:
            yield request
            waits.append(env.now - arrival)
            yield env.timeout(float(service))
    for arrival, service in zip(arrivals, services):
        env.process(customer(arrival, service))
    env.run()
    return np.array(waits)
assert np.allclose(simulate([0, 1], [3, 2], 1), [0, 2])
assert np.allclose(simulate([0, 1], [3, 2], 2), [0, 0])

# %%
# ### 3. 用相同顾客情景比较
records = []
for seed in range(40):
    day_rng = np.random.default_rng(1000 + seed)
    arrivals = np.cumsum(day_rng.choice(calibration.interarrival, 120))
    services = day_rng.choice(calibration.service, 120)
    for capacity in [2, 3]:
        w = simulate(arrivals, services, capacity)
        records.append({'seed': seed, 'windows': capacity, 'wait_mean': w.mean(), 'wait_p90': np.quantile(w, 0.9), 'staff_cost': capacity * 160})
comparison = pd.DataFrame(records)
summary = comparison.groupby('windows')[['wait_mean', 'wait_p90', 'staff_cost']].mean()
actual_arrivals = np.cumsum(holdout.interarrival.to_numpy())
held_wait = {str(c): float(simulate(actual_arrivals, holdout.service, c).mean()) for c in [2, 3]}
comparison.to_csv(OUT_DIR / 'daily_comparison.csv', index=False)
summary.wait_mean.plot.bar(color=['#2a6f97', '#e9a23b'])
plt.ylabel('每位顾客平均等待（分钟）')
plt.title('共同情景下的窗口配置')
plt.tight_layout()
plt.savefig(OUT_DIR / 'waiting.png')
plt.close()

# %%
# ### 4. 把等待变化翻译成取舍
reduction = float(summary.loc[2, 'wait_mean'] - summary.loc[3, 'wait_mean'])
metrics = {'calibrated_interarrival_mean': arrival_mean, 'calibrated_service_mean': service_mean, 'mean_wait_reduction_minutes': reduction, 'extra_staff_cost': 160, 'heldout_wait_minutes': held_wait}
recommendation = f'从两个增至三个窗口，在校准情景中平均减少等待 {reduction:.2f} 分钟/人，增加160元/班。是否值得取决于顾客时间价值和服务目标。留出记录结果为 {held_wait}；真实使用前须收集真实到达/服务记录。'

# %%
# ### 保存可核验的证据
import json
(OUT_DIR / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
(OUT_DIR / '决策建议.md').write_text('# 本次计算的决策建议\n\n' + recommendation + '\n\n## 实际指标\n```json\n' + json.dumps(metrics, ensure_ascii=False, indent=2) + '\n```\n\n这些结论只适用于本例显式假设；目标权重和风险偏好由人决定。', encoding='utf-8')
print(metrics)