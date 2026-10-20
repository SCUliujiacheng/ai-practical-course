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
# ## 1. 事实集合与动作三元组
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

from collections import deque
def action(name,pre,add,delete):
    return dict(name=name,pre=set(pre),add=set(add),delete=set(delete))
actions=[action("take-key",["at-hall","key-hall"],["has-key"],["key-hall"]),
         action("open-door",["at-hall","has-key"],["door-open"],[]),
         action("take-package",["at-hall","package-hall"],["carrying"],["package-hall"]),
         action("enter-room",["at-hall","door-open"],["at-room"],["at-hall"]),
         action("leave-room",["at-room","door-open"],["at-hall"],["at-room"]),
         action("deliver",["at-room","carrying"],["delivered"],["carrying"])]
initial_state=frozenset(["at-hall","key-hall","package-hall"]); target={"delivered"}
def apply_action(state,rule):
    if not rule["pre"]<=set(state): raise ValueError("前提未满足："+rule["name"])
    return frozenset((set(state)-rule["delete"])|rule["add"])
print("初始事实",sorted(initial_state))


# %%
# ## 2. 前向广度优先搜索
def plan_actions(start,goal,rules):
    queue=deque([(frozenset(start),[])]);visited={frozenset(start)}
    while queue:
        state,plan=queue.popleft()
        if set(goal)<=set(state): return plan,len(visited)
        for rule in rules:
            if rule["pre"]<=set(state):
                nxt=apply_action(state,rule)
                if nxt not in visited:
                    visited.add(nxt);queue.append((nxt,plan+[rule["name"]]))
    return None,len(visited)
plan,visited_count=plan_actions(initial_state,target,actions)
assert plan is not None and len(plan)==5
impossible,_=plan_actions(initial_state,{"teleported"},actions)
assert impossible is None
print("动作计划",plan)


# %%
# ## 3. 按原任务写独立执行器
def validate_business_plan(sequence):
    location="hall";key=False;door=False;carrying=False;delivered=False;key_on_floor=True;package_on_floor=True
    trace=[]
    for step,name in enumerate(sequence):
        ok=True
        if name=="take-key":
            ok=location=="hall" and key_on_floor
            if ok:key=True;key_on_floor=False
        elif name=="open-door":
            ok=location=="hall" and key
            if ok:door=True
        elif name=="take-package":
            ok=location=="hall" and package_on_floor
            if ok:carrying=True;package_on_floor=False
        elif name=="enter-room":
            ok=location=="hall" and door
            if ok:location="room"
        elif name=="leave-room":
            ok=location=="room" and door
            if ok:location="hall"
        elif name=="deliver":
            ok=location=="room" and carrying
            if ok:delivered=True;carrying=False
        else:ok=False
        trace.append(dict(step=step,action=name,valid=ok,location=location,key=key,door=door,carrying=carrying,delivered=delivered))
        if not ok:return False,trace
    return delivered,trace
valid,trace=validate_business_plan(plan)
assert valid
assert not validate_business_plan(["open-door","enter-room","deliver"])[0]
assert not validate_business_plan(["invent-action"])[0]


# %%
# ## 4. 对照PDDL，检查离线错误草稿
import copy
draft=copy.deepcopy(actions)
draft[1]["pre"].remove("has-key")
draft_plan,_=plan_actions(initial_state,target,draft)
draft_ok,draft_trace=validate_business_plan(draft_plan)
assert not draft_ok and len(draft_plan)==4
(OUT_DIR/"离线错误草稿.json").write_text(json.dumps(draft,ensure_ascii=False,default=sorted,indent=2),encoding="utf-8")
(OUT_DIR/"计划.txt").write_text("\n".join(plan),encoding="utf-8")
pd.DataFrame(trace).to_csv(OUT_DIR/"状态追踪.csv",index=False,encoding="utf-8-sig")
pd.DataFrame(draft_trace).to_csv(OUT_DIR/"错误草稿检查.csv",index=False,encoding="utf-8-sig")
plt.barh(range(len(plan)),[1]*len(plan),left=range(len(plan)),color="#498a9c")
plt.yticks(range(len(plan)),plan);plt.xlabel("动作步")
plt.tight_layout();plt.savefig(OUT_DIR/"动作序列.png");plt.close()
save_results({"plan":plan,"action_count":len(plan),"visited_states":visited_count,
              "business_validator_pass":valid,"draft_plan":draft_plan,"draft_validator_pass":draft_ok,
              "external_pddl_solver_run":False},
             f"采用{len(plan)}步计划：{' → '.join(plan)}。独立业务执行检查通过。"
             f"漏写钥匙前提的草稿产生{len(draft_plan)}步计划，但独立检查拒绝；求解器只能解已写出的模型，不能自动补齐现实需求。")
