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
# ## 1. 定义地图、动作和碰撞
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

import heapq
from collections import deque
height=5;width=5;walls={(1,1),(1,3),(3,1),(3,3)}
starts=[(2,0),(2,4),(0,2)]; goals=[(2,4),(2,0),(4,2)]
def moves(pos):
    return [(pos[0]+dr,pos[1]+dc) for dr,dc in [(0,0),(1,0),(-1,0),(0,1),(0,-1)]
            if 0<=pos[0]+dr<height and 0<=pos[1]+dc<width and (pos[0]+dr,pos[1]+dc) not in walls]
def at(path,t): return path[min(t,len(path)-1)]
def first_conflict(paths):
    end=max(map(len,paths))
    for t in range(end):
        for i in range(len(paths)):
            for j in range(i+1,len(paths)):
                if at(paths[i],t)==at(paths[j],t): return ("v",i,j,t,at(paths[i],t))
                if t>0 and at(paths[i],t-1)==at(paths[j],t) and at(paths[j],t-1)==at(paths[i],t):
                    return ("e",i,j,t,at(paths[i],t-1),at(paths[i],t))
    return None
(DATA_DIR/"地图.json").write_text(json.dumps(dict(height=height,width=width,walls=list(walls),starts=starts,goals=goals)),encoding="utf-8")


# %%
# ## 2. 时空A*：同一地点在不同时间是不同状态
def low_level(origin,target,constraints,horizon=30):
    vertices={(c[1],c[2]) for c in constraints if c[0]=="v"}
    edges={(c[1],c[2],c[3]) for c in constraints if c[0]=="e"}
    if (0,origin) in vertices: return None
    def h(p): return abs(p[0]-target[0])+abs(p[1]-target[1])
    count=itertools.count(); queue=[(h(origin),0,next(count),origin,[origin])];seen={(origin,0)}
    while queue:
        _,t,_,pos,path=heapq.heappop(queue)
        future_block=any(tt>=t and cell==target for tt,cell in vertices)
        if pos==target and not future_block: return path
        if t>=horizon: continue
        for nxt in moves(pos):
            tt=t+1
            if (tt,nxt) in vertices or (tt,pos,nxt) in edges: continue
            state=(nxt,tt)
            if state in seen: continue
            seen.add(state)
            heapq.heappush(queue,(tt+h(nxt),tt,next(count),nxt,path+[nxt]))
    return None
independent=[low_level(s,g,[]) for s,g in zip(starts,goals)]
assert all(p is not None for p in independent)
assert first_conflict(independent) is not None


# %%
# ## 3. 先规划的车占用后续车辆的时空资源
def prioritized(order,horizon=30):
    result=[None]*len(starts); fixed=[]
    for agent in order:
        constraints=[]
        for old in fixed:
            for t in range(horizon+1):
                constraints.append(("v",t,at(old,t)))
                if t>0: constraints.append(("e",t,at(old,t),at(old,t-1)))
        path=low_level(starts[agent],goals[agent],constraints,horizon)
        if path is None: return None
        result[agent]=path;fixed.append(path)
    return result
priority_paths=prioritized([0,1,2])
assert priority_paths is not None and first_conflict(priority_paths) is None


# %%
# ## 4. CBS：发现冲突，再分支限制其中一辆车
def cbs(horizon=30,node_limit=2000):
    root_paths=[low_level(s,g,[],horizon) for s,g in zip(starts,goals)]
    if any(p is None for p in root_paths): return "UNKNOWN_HORIZON",None,0
    counter=itertools.count(); queue=[];visited=set(); expanded=0
    cost=lambda paths:sum(len(p)-1 for p in paths)
    heapq.heappush(queue,(cost(root_paths),next(counter),[[] for _ in starts],root_paths))
    while queue and expanded<node_limit:
        _,_,restrictions,paths=heapq.heappop(queue); expanded+=1
        issue=first_conflict(paths)
        if issue is None: return "OPTIMAL_WITHIN_TIME_MODEL",paths,expanded
        kind,i,j,t,*positions=issue
        for agent in (i,j):
            new=[list(c) for c in restrictions]
            if kind=="v": added=("v",t,positions[0])
            else:
                u,v=positions
                added=("e",t,u,v) if agent==i else ("e",t,v,u)
            new[agent].append(added)
            signature=tuple(tuple(sorted(cs)) for cs in new)
            if signature in visited: continue
            visited.add(signature)
            updated=low_level(starts[agent],goals[agent],new[agent],horizon)
            if updated is not None:
                child=list(paths);child[agent]=updated
                heapq.heappush(queue,(cost(child),next(counter),new,child))
    return "UNKNOWN_LIMIT_OR_HORIZON",None,expanded
cbs_status,cbs_paths,cbs_expanded=cbs()
assert cbs_paths is not None
def validate_paths(paths):
    if paths is None or len(paths)!=len(starts): return False
    if any(not p or p[0]!=s or p[-1]!=g for p,s,g in zip(paths,starts,goals)): return False
    # 重新按业务规则回放，不调用搜索使用的 moves/first_conflict。
    for path in paths:
        for row,col in path:
            if not (0<=row<height and 0<=col<width) or (row,col) in walls: return False
        for u,v in zip(path,path[1:]):
            if abs(u[0]-v[0])+abs(u[1]-v[1])>1: return False
    previous=None
    for t in range(max(map(len,paths))):
        current=[p[t] if t<len(p) else p[-1] for p in paths]
        if len(set(current))!=len(current): return False
        if previous is not None:
            for i in range(len(paths)):
                for j in range(i+1,len(paths)):
                    if current[i]==previous[j] and current[j]==previous[i]: return False
        previous=current
    return True
assert validate_paths(cbs_paths) and validate_paths(priority_paths)
assert sum(len(p)-1 for p in cbs_paths)<=sum(len(p)-1 for p in priority_paths)
assert cbs(node_limit=0)[0].startswith("UNKNOWN")


# %%
# ## 5. 保存逐时刻轨迹并解释目标
rows=[]
for method,paths in [("优先规划",priority_paths),("CBS",cbs_paths)]:
    for t in range(max(map(len,paths))):
        for agent,p in enumerate(paths): rows.append(dict(method=method,time=t,train=agent,row=at(p,t)[0],col=at(p,t)[1]))
pd.DataFrame(rows).to_csv(OUT_DIR/"列车时刻表.csv",index=False,encoding="utf-8-sig")
fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,label,paths in zip(axes,["优先规划","CBS"],[priority_paths,cbs_paths]):
    for a,p in enumerate(paths): ax.plot([q[1] for q in p],[q[0] for q in p],"o-",label=f"车{a}",alpha=.7)
    for r,c in walls: ax.add_patch(plt.Rectangle((c-.45,r-.45),.9,.9,color="gray"))
    ax.set(xlim=(-.5,4.5),ylim=(4.5,-.5),title=label);ax.legend();ax.grid()
plt.tight_layout();plt.savefig(OUT_DIR/"多车路线.png");plt.close()
# 平面路线可能交叉；是否碰撞取决于时间。用时间表看清等待和到站占用。
fig,ax=plt.subplots(figsize=(10,2.5));ax.axis("off")
timeline=range(max(map(len,cbs_paths)))
positions=[[f"({at(p,t)[0]},{at(p,t)[1]})" for t in timeline] for p in cbs_paths]
table=ax.table(cellText=positions,rowLabels=[f"车{a}" for a in range(len(cbs_paths))],
               colLabels=[f"t={t}" for t in timeline],loc="center",cellLoc="center")
table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,1.7)
ax.set_title("CBS逐时刻位置：到站后继续占用终点")
plt.tight_layout();plt.savefig(OUT_DIR/"逐时刻位置.png");plt.close()
cost=lambda ps:sum(len(p)-1 for p in ps)
save_results({"cbs_status":cbs_status,"cbs_sum_cost":cost(cbs_paths),"priority_sum_cost":cost(priority_paths),
              "cbs_makespan":max(len(p)-1 for p in cbs_paths),"cbs_high_level_nodes":cbs_expanded,"collision_free":True},
             f"CBS方案总到达时间{cost(cbs_paths)}，优先方案{cost(priority_paths)}；"
             f"CBS高层展开{cbs_expanded}个节点。逐步合法性、同点冲突、对向交换和终点占用均检查通过。"
             "最优结论限定在本离散动作、总到达时间目标和30步时间窗模型内；实际列车的速度、制动和安全间隔未建模。")
