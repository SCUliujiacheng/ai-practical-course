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
# ## 1. 先列出每个团队的合法选项
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
jobs = [dict(name="光学组",duration=2,rooms=[0],starts=[0,1,2],members={"Li","Wu"}),
        dict(name="编程组",duration=1,rooms=[0,1],starts=[0,1,2,3],members={"Sun"}),
        dict(name="测试组",duration=2,rooms=[0,1],starts=[1,2,3,4],members={"Li"}),
        dict(name="阅读组",duration=1,rooms=[1],starts=[0,1,2,3],members={"Qian"}),
        dict(name="装配组",duration=2,rooms=[0,1],starts=[0,1,2,3,4],members={"Zhou"}),
        dict(name="演示组",duration=1,rooms=[0],starts=[3,4,5],members={"Sun"})]
def options(job):
    return [(room,start,start+job["duration"]) for room in job["rooms"] for start in job["starts"]]
def conflict(a, b, job_a, job_b):
    overlap = max(a[1],b[1]) < min(a[2],b[2])
    shared_resource = a[0] == b[0] or bool(job_a["members"] & job_b["members"])
    return overlap and shared_resource
assert not conflict((0,0,1),(0,1,2),jobs[0],jobs[1])
(DATA_DIR / "预约需求.json").write_text(json.dumps(jobs,ensure_ascii=False,default=list,indent=2),encoding="utf-8")


# %%
# ## 2. 回溯寻找第一份合法计划
def backtrack_booking(requests):
    order = sorted(range(len(requests)),key=lambda i:len(options(requests[i])))
    plan = {}
    def visit(k):
        if k == len(order): return dict(plan)
        i = order[k]
        for choice in options(requests[i]):
            if all(not conflict(choice,other,requests[i],requests[j]) for j,other in plan.items()):
                plan[i] = choice
                found = visit(k+1)
                if found is not None: return found
                del plan[i]
        return None
    return visit(0)
baseline = backtrack_booking(jobs)
assert baseline is not None


# %%
# ## 3. 用CP-SAT选择互不冲突的选项
def solve_booking(requests):
    model=cp_model.CpModel(); candidates=[options(j) for j in requests]; variables={}
    for i,choices in enumerate(candidates):
        variables[i] = [model.new_bool_var(f"team{i}_option{k}") for k in range(len(choices))]
        model.add_exactly_one(variables[i])
    for i in range(len(requests)):
        for j in range(i+1,len(requests)):
            for k,a in enumerate(candidates[i]):
                for l,b in enumerate(candidates[j]):
                    if conflict(a,b,requests[i],requests[j]): model.add(variables[i][k]+variables[j][l] <= 1)
    horizon=max((c[2] for choices in candidates for c in choices),default=0)
    end=model.new_int_var(0,horizon,"end")
    for i,choices in enumerate(candidates):
        for k,c in enumerate(choices): model.add(end>=c[2]).only_enforce_if(variables[i][k])
    model.minimize(end)
    solver=cp_model.CpSolver(); solver.parameters.max_time_in_seconds=5; solver.parameters.num_search_workers=1
    state=solver.solve(model)
    result=({i:next(c for k,c in enumerate(choices) if solver.value(variables[i][k])) for i,choices in enumerate(candidates)}
            if state in (cp_model.OPTIMAL,cp_model.FEASIBLE) else None)
    return solver.status_name(state),result
def check_booking(plan,requests):
    if plan is None or set(plan)!=set(range(len(requests))): return False
    # 直接核对原始需求，不复用求解器生成选项或冲突约束的函数。
    for i,(room,start,end) in plan.items():
        request=requests[i]
        if room not in request['rooms'] or start not in request['starts']: return False
        if end-start!=request['duration']: return False
    for i in plan:
        for j in plan:
            if i>=j: continue
            a,b=plan[i],plan[j]
            if a[1]<b[2] and b[1]<a[2]:
                if a[0]==b[0]: return False
                if any(member in requests[j]['members'] for member in requests[i]['members']): return False
    return True
status,plan=solve_booking(jobs)
assert plan is not None, f"未获得预约表：{status}；UNKNOWN不等于无解。"
assert check_booking(plan,jobs) and check_booking(baseline,jobs)
corrupted=dict(plan); room,start,end=corrupted[0]; corrupted[0]=(room,start,end+1)
assert not check_booking(corrupted,jobs)
impossible=[dict(name=str(i),duration=1,rooms=[0],starts=[0],members={str(i)}) for i in range(2)]
bad_status,bad_plan=solve_booking(impossible)
assert bad_status=="INFEASIBLE" and bad_plan is None


# %%
# ## 4. 保存预约表与冲突反例
table=pd.DataFrame([dict(team=jobs[i]["name"],room=c[0],start=c[1],end=c[2]) for i,c in plan.items()])
table.to_csv(OUT_DIR / "预约表.csv",index=False,encoding="utf-8-sig")
for i,c in plan.items():
    plt.barh(c[0],c[2]-c[1],left=c[1],height=.5,alpha=.7)
    plt.text((c[1]+c[2])/2,c[0],jobs[i]["name"],ha="center",va="center",fontsize=8)
plt.yticks([0,1],["实验室0","实验室1"]); plt.xlabel("时段")
plt.tight_layout(); plt.savefig(OUT_DIR / "预约安排.png"); plt.close()
end=max(c[2] for c in plan.values()); baseline_end=max(c[2] for c in baseline.values())
save_results({"status":status,"last_end":end,"backtracking_last_end":baseline_end,
              "constraint_check":True,"infeasible_example":bad_status},
             f"采用预约表中的方案，最后在时段{end}结束，回溯基线为{baseline_end}；状态{status}。"
             "房间重叠和成员分身均独立检查通过。若可用时间变化，应先更新需求再求解。")
