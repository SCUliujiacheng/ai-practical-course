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
# ## 1. 人员时间表与硬条件
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
availability=np.array([[1,1,1,1],[1,1,0,1],[1,0,1,1],[1,1,1,0],
                       [0,1,1,1],[1,1,0,1],[1,0,1,1],[0,1,1,1]],int)
preferences=np.array([[1,0,1,0],[1,1,0,0],[0,0,1,1],[1,0,1,0],
                      [0,1,0,1],[1,0,0,1],[0,0,1,1],[0,1,1,0]],int)
skilled={0,1,2};demand=[3,3,3,3]
pd.DataFrame(availability,columns=[f"时段{s}" for s in range(4)]).to_csv(DATA_DIR/"可用时间.csv",encoding="utf-8-sig")


# %%
# ## 2. 用二元变量表达排班
def schedule(available,preferred,qualified,needed):
    workers,slots=available.shape;model=cp_model.CpModel()
    x={(w,s):model.new_bool_var(f"w{w}s{s}") for w in range(workers) for s in range(slots)}
    for w in range(workers):
        for s in range(slots):model.add(x[w,s]<=int(available[w,s]))
        model.add(sum(x[w,s] for s in range(slots))<=2)
        for s in range(slots-2):model.add(sum(x[w,k] for k in range(s,s+3))<=2)
    for s in range(slots):
        model.add(sum(x[w,s] for w in range(workers))==int(needed[s]))
        model.add(sum(x[w,s] for w in qualified)>=1)
    loads=[model.new_int_var(0,2,f"load{w}") for w in range(workers)]
    for w in range(workers):model.add(loads[w]==sum(x[w,s] for s in range(slots)))
    maximum=model.new_int_var(0,2,"maximum");minimum=model.new_int_var(0,2,"minimum")
    model.add_max_equality(maximum,loads);model.add_min_equality(minimum,loads)
    pref=sum(int(preferred[w,s])*x[w,s] for w in range(workers) for s in range(slots))
    priority_weight=workers*slots+1
    model.minimize(priority_weight*(maximum-minimum)-pref)
    solver=cp_model.CpSolver();solver.parameters.max_time_in_seconds=8;solver.parameters.num_search_workers=1
    state=solver.solve(model)
    plan=(np.array([[solver.value(x[w,s]) for s in range(slots)] for w in range(workers)],int)
          if state in (cp_model.OPTIMAL,cp_model.FEASIBLE) else None)
    return solver.status_name(state),plan
status,roster=schedule(availability,preferences,skilled,demand)
assert roster is not None, f"未获得排班：{status}；UNKNOWN不能解释为已证明无解。"


# %%
# ## 3. 独立检查原始业务约束
def check_roster(roster,available,qualified,needed):
    if roster is None or roster.shape!=available.shape:return False
    if not np.isin(roster,[0,1]).all() or np.any(roster>available):return False
    if not np.array_equal(roster.sum(axis=0),np.array(needed)):return False
    if np.any(roster.sum(axis=1)>2):return False
    if np.any(roster[list(qualified)].sum(axis=0)<1):return False
    return all(roster[w,s:s+3].sum()<=2 for w in range(len(roster)) for s in range(roster.shape[1]-2))
assert check_roster(roster,availability,skilled,demand)
no_skill=availability.copy();no_skill[list(skilled),0]=0
bad_status,bad_roster=schedule(no_skill,preferences,skilled,demand)
assert bad_status=="INFEASIBLE" and bad_roster is None
broken=roster.copy();broken[:,0]=0
assert not check_roster(broken,availability,skilled,demand)


# %%
# ## 4. 保存排班并说明公平定义
table=pd.DataFrame(roster,index=[f"志愿者{w}" for w in range(len(roster))],columns=[f"时段{s}" for s in range(roster.shape[1])])
table["总班数"]=roster.sum(axis=1);table["急救资格"]=[w in skilled for w in range(len(roster))]
table.to_csv(OUT_DIR/"排班表.csv",encoding="utf-8-sig")
plt.imshow(roster,cmap="Blues",vmin=0,vmax=1,aspect="auto");plt.yticks(range(8),table.index);plt.xticks(range(4),[f"时段{s}" for s in range(4)])
plt.title("深色表示上岗");plt.tight_layout();plt.savefig(OUT_DIR/"排班.png");plt.close()
spread=int(np.ptp(roster.sum(axis=1)));preferred_count=int((roster*preferences).sum())
save_results({"status":status,"workload_spread":spread,"preferred_assignments":preferred_count,
              "coverage":roster.sum(axis=0).tolist(),"all_constraints_pass":True,"infeasible_test":bad_status},
             f"推荐排班表，4段覆盖{roster.sum(axis=0).tolist()}，总班数差{spread}，满足{preferred_count}个偏好。"
             f"状态{status}；如果急救员在首段都不可用，需求已被证明无解，需要改需求或增加合格人员。")
