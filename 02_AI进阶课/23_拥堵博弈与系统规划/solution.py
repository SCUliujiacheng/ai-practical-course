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
# ## 1. 耗时依赖同路人数
import json
import itertools
import math
import numpy as np
import pandas as pd

def save_results(metrics, recommendation):
    """只负责保存实际计算结果；决策逻辑在下文。"""
    def convert(value):
        return value.item() if hasattr(value, "item") else str(value)
    (OUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=convert), encoding="utf-8")
    (OUT_DIR / "决策建议.md").write_text(
        "# 本次运行的决策建议\n\n" + recommendation +
        "\n\n数据为本例明确给定的教学场景；结论仅对模型、目标与已检查条件成立。\n",
        encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=convert))
    print(recommendation)

players=6
def latency(route,count):return float(count if route==0 else 5)
def individual_costs(profile):
    counts=[profile.count(r) for r in [0,1]]
    return np.array([latency(r,counts[r]) for r in profile])
def potential(profile):
    return sum(sum(latency(r,k) for k in range(1,profile.count(r)+1)) for r in [0,1])
def equilibrium(profile):
    current=individual_costs(profile)
    for i in range(len(profile)):
        alternative=list(profile);alternative[i]=1-profile[i]
        if individual_costs(alternative)[i]<current[i]-1e-9:return False
    return True
assert individual_costs([0]*6).tolist()==[6]*6


# %%
# ## 2. 逐个参与者严格改进
def best_response(start,limit=100):
    profile=list(start);trace=[dict(step=0,profile=list(profile),potential=potential(profile),total=float(individual_costs(profile).sum()))]
    for step in range(limit):
        changed=False
        for i in range(len(profile)):
            alternative=list(profile);alternative[i]=1-profile[i]
            if individual_costs(alternative)[i]<individual_costs(profile)[i]-1e-9:
                before=potential(profile);profile=alternative
                assert potential(profile)<before
                trace.append(dict(step=len(trace),profile=list(profile),potential=potential(profile),total=float(individual_costs(profile).sum())))
                changed=True;break
        if not changed:return "EQUILIBRIUM",profile,trace
    return "LIMIT",profile,trace
status,selfish,trace=best_response([0]*players)
assert status=="EQUILIBRIUM" and equilibrium(selfish)


# %%
# ## 3. 系统优化与均衡分别枚举
profiles=[list(p) for p in itertools.product([0,1],repeat=players)]
optimal=min(profiles,key=lambda p:individual_costs(p).sum())
equilibria=[p for p in profiles if equilibrium(p)]
optimum=float(individual_costs(optimal).sum())
equilibrium_costs=[float(individual_costs(p).sum()) for p in equilibria]
assert len(equilibria)>0 and optimum<=min(equilibrium_costs)
worst_ratio=max(equilibrium_costs)/optimum
forced_extra=[i for i in range(players) if individual_costs(optimal)[i]>individual_costs(selfish)[i]]


# %%
# ## 4. 输出协调收益和个体代价
pd.DataFrame([dict(person=i,selfish_route=selfish[i],planned_route=optimal[i],
                   selfish_time=individual_costs(selfish)[i],planned_time=individual_costs(optimal)[i]) for i in range(players)]).to_csv(OUT_DIR/"个体对照.csv",index=False,encoding="utf-8-sig")
pd.DataFrame(trace).to_csv(OUT_DIR/"最佳回应轨迹.csv",index=False,encoding="utf-8-sig")
split_costs=[float(individual_costs([0]*n+[1]*(players-n)).sum()) for n in range(players+1)]
plt.plot(range(players+1),split_costs,"o-");plt.xlabel("走捷径的人数");plt.ylabel("系统总耗时")
plt.tight_layout();plt.savefig(OUT_DIR/"拥堵取舍.png");plt.close()
save_results({"equilibrium_profile":selfish,"equilibrium_total":float(individual_costs(selfish).sum()),
              "optimal_profile":optimal,"optimal_total":optimum,"equilibrium_count":len(equilibria),
              "worst_equilibrium_ratio":worst_ratio,"potential_trace":[r["potential"] for r in trace]},
             f"系统安排总耗时{optimum:g}，当前自发稳定配置为{individual_costs(selfish).sum():g}；"
             f"需要协调的路线分配为{optimal}。相对该初始稳定配置变慢的参与者编号{forced_extra}，应解释或处理个体取舍。")
