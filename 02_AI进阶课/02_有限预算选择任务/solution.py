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
# ## 1. 明确输入与评分来源
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

from ortools.sat.python import cp_model
tasks = [{"name": f"任务{i}", "cost": c, "hours": h, "value": v}
         for i, (c, h, v) in enumerate([(3,2,6),(4,3,7),(2,2,4),(5,4,10),(3,1,5),
                                       (6,5,11),(2,1,3),(4,2,8),(3,3,5),(5,3,9)])]
budget, hours = 16, 12
dependencies = [(3, 0), (5, 2), (7, 1)]
pd.DataFrame(tasks).to_csv(DATA_DIR / "候选任务.csv", index=False, encoding="utf-8-sig")
def valid_choice(chosen, jobs, money, time_limit, deps):
    if len(set(chosen)) != len(chosen): return False
    chosen = set(chosen)
    return (all(0 <= i < len(jobs) for i in chosen)
            and sum(jobs[i]["cost"] for i in chosen) <= money
            and sum(jobs[i]["hours"] for i in chosen) <= time_limit
            and all(a not in chosen or b in chosen for a, b in deps))
assert not valid_choice([0,0],tasks,100,100,dependencies)


# %%
# ## 2. 单预算背包：理解动态规划
def knapsack_value(jobs, cap):
    dp = [0] * (cap + 1)
    for job in jobs:
        for b in range(cap, job["cost"] - 1, -1):
            dp[b] = max(dp[b], dp[b-job["cost"]] + job["value"])
    return dp[-1]
unconstrained_value = knapsack_value(tasks, budget)
assert knapsack_value([{"cost": 2, "value": 3}], 4) == 3
print("只有预算时的价值上界：", unconstrained_value)


# %%
# ## 3. CP-SAT 同时处理多个约束
def optimize_tasks(jobs, money, time_limit, deps, require_all=False):
    model = cp_model.CpModel()
    x = [model.new_bool_var(f"pick_{i}") for i in range(len(jobs))]
    model.add(sum(j["cost"] * x[i] for i,j in enumerate(jobs)) <= money)
    model.add(sum(j["hours"] * x[i] for i,j in enumerate(jobs)) <= time_limit)
    for a,b in deps: model.add(x[a] <= x[b])
    if require_all:
        for variable in x: model.add(variable == 1)
    model.maximize(sum(j["value"] * x[i] for i,j in enumerate(jobs)))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5
    solver.parameters.num_search_workers = 1
    status = solver.solve(model)
    chosen = ([i for i,var in enumerate(x) if solver.value(var)]
              if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None)
    return solver.status_name(status), chosen
status, chosen = optimize_tasks(tasks, budget, hours, dependencies)
assert chosen is not None, f"未获得方案：{status}。UNKNOWN表示尚未找到或证明，不能解释为无解。"
assert valid_choice(chosen, tasks, budget, hours, dependencies)
# 10个任务只有1024种组合，用独立穷举核对小实例。
feasible = [list(i for i in range(len(tasks)) if bits[i])
            for bits in itertools.product([0,1], repeat=len(tasks))
            if valid_choice([i for i,b in enumerate(bits) if b], tasks, budget, hours, dependencies)]
value = sum(tasks[i]["value"] for i in chosen)
exact = max(sum(tasks[i]["value"] for i in c) for c in feasible)
assert value == exact and value <= unconstrained_value
impossible, _ = optimize_tasks(tasks, 0, 0, dependencies, require_all=True)
assert impossible == "INFEASIBLE"


# %%
# ## 4. 与简单选择法比较并保存计划
greedy = []
for i in sorted(range(len(tasks)), key=lambda i: tasks[i]["value"]/tasks[i]["cost"], reverse=True):
    if valid_choice(greedy+[i], tasks, budget, hours, dependencies): greedy.append(i)
greedy_value = sum(tasks[i]["value"] for i in greedy)
pd.DataFrame([tasks[i] for i in chosen]).to_csv(OUT_DIR / "任务计划.csv", index=False, encoding="utf-8-sig")
plt.bar(["简单排序", "约束优化"], [greedy_value, value]); plt.ylabel("假设价值总分")
plt.tight_layout(); plt.savefig(OUT_DIR / "任务选择.png"); plt.close()
save_results({"status":status,"selected":chosen,"value":value,"greedy_value":greedy_value,
              "cost":sum(tasks[i]["cost"] for i in chosen),"hours":sum(tasks[i]["hours"] for i in chosen),
              "infeasible_test":impossible},
             f"推荐任务{chosen}，价值{value}，基线{greedy_value}；求解状态{status}。"
             "已逐条检查预算、工时和依赖，并用1024个组合核对本小实例最优值。评分变化时需要重算。")
