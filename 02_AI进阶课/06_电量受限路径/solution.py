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
# ## 1. 一条边包含两个不同维度
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
import networkx as nx
road_data=[("S","A",1,5),("A","G",1,5),("S","B",2,2),("B","C",2,2),
           ("C","G",2,2),("S","D",3,1),("D","G",4,3),("B","D",2,1)]
roads={}
for u,v,time,energy in road_data:
    roads.setdefault(u,[]).append((v,time,energy)); roads.setdefault(v,[]).append((u,time,energy))
battery=6; start="S"; goal="G"
graph=nx.Graph()
for u,v,time,energy in road_data: graph.add_edge(u,v,time=time,energy=energy)
fastest=nx.shortest_path(graph,start,goal,weight="time")
pd.DataFrame(road_data,columns=["from","to","time","energy"]).to_csv(DATA_DIR / "道路.csv",index=False)


# %%
# ## 2. 扩展状态：位置相同，剩余电量不同
def battery_route(network,origin,target,capacity):
    queue=[(0,origin,capacity,[origin])]; best={(origin,capacity):0}; expanded=0
    while queue:
        elapsed,node,remaining,path=heapq.heappop(queue)
        if elapsed != best.get((node,remaining)): continue
        expanded+=1
        if node==target: return path,elapsed,capacity-remaining,expanded
        for nxt,duration,energy in network.get(node,[]):
            if energy>remaining: continue
            state=(nxt,remaining-energy); candidate=elapsed+duration
            if candidate<best.get(state,float("inf")):
                best[state]=candidate
                heapq.heappush(queue,(candidate,nxt,remaining-energy,path+[nxt]))
    return None,None,None,expanded
route,time,used,expanded=battery_route(roads,start,goal,battery)
assert route is not None


# %%
# ## 3. 用原始道路独立验证
def path_totals(path,network):
    if path is None: return None
    total_time=0;total_energy=0
    for u,v in zip(path,path[1:]):
        matching=[edge for edge in network.get(u,[]) if edge[0]==v]
        if not matching: return None
        _,dt,de=matching[0];total_time+=dt;total_energy+=de
    return total_time,total_energy
assert path_totals(route,roads)==(time,used) and used<=battery
simple_paths=list(nx.all_simple_paths(graph,start,goal))
feasible=[(path_totals(p,roads)[0],p) for p in simple_paths if path_totals(p,roads)[1]<=battery]
assert time==min(t for t,p in feasible)
assert battery_route(roads,start,goal,1)[0] is None
fastest_time,fastest_energy=path_totals(fastest,roads)


# %%
# ## 4. 区分不可行基线与有效方案
table=pd.DataFrame([dict(method="忽略电量",path="→".join(fastest),time=fastest_time,energy=fastest_energy,feasible=fastest_energy<=battery),
                    dict(method="电量约束",path="→".join(route),time=time,energy=used,feasible=True)])
table.to_csv(OUT_DIR / "路线对照.csv",index=False,encoding="utf-8-sig")
pos={"S":(0,0),"A":(1,1),"B":(1,0),"C":(2,0),"D":(1,-1),"G":(3,0)}
nx.draw(graph,pos,with_labels=True,node_color="#a9d5dc",node_size=900)
nx.draw_networkx_edges(graph,pos,edgelist=list(zip(route,route[1:])),edge_color="#df7843",width=3)
plt.title("满足电量限制的路线");plt.tight_layout();plt.savefig(OUT_DIR / "路线.png");plt.close()
save_results({"route":route,"time":time,"energy":used,"battery":battery,
              "unconstrained_time":fastest_time,"unconstrained_energy":fastest_energy,"expanded_states":expanded},
             f"推荐{'→'.join(route)}，时间{time}、电量{used}/{battery}。"
             f"忽略电量的最快路线耗电{fastest_energy}，可行性为{fastest_energy<=battery}。电池容量或耗电估计改变时重算。")
