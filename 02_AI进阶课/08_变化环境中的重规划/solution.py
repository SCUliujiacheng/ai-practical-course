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
# ## 1. 建立道路和公开的事件机制
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

import networkx as nx
initial=nx.Graph()
initial.add_weighted_edges_from([("S","A",1),("A","G",1),("A","B",1),("B","G",3),("B","C",1),("C","G",1),("S","B",3)])
events=[dict(time=1,kind="close",u="A",v="G"),dict(time=2,kind="cost",u="B",v="C",value=9)]
def observe(graph,clock,applied):
    messages=[]
    for k,event in enumerate(events):
        if k in applied or event["time"]>clock: continue
        if event["kind"]=="close": graph.remove_edge(event["u"],event["v"])
        else: graph[event["u"]][event["v"]]["weight"]=event["value"]
        applied.add(k);messages.append(event["kind"]+":"+event["u"]+"-"+event["v"])
    return messages
old_path=nx.shortest_path(initial,"S","G",weight="weight")
(DATA_DIR/"事件.json").write_text(json.dumps(events,indent=2),encoding="utf-8")


# %%
# ## 2. 三种反馈规则共用一个执行环境
def execute(policy):
    graph=initial.copy();applied=set();clock=0;node="S";planned=list(old_path);trace=[];changes=0
    for step in range(20):
        news=observe(graph,clock,applied)
        if node=="G": return dict(policy=policy,success=True,time=clock,changes=changes,trace=trace)
        legal=all(graph.has_edge(u,v) for u,v in zip(planned,planned[1:]))
        should_replan=policy=="rolling" or (policy=="repair_blocked" and not legal)
        if should_replan:
            try: replacement=nx.shortest_path(graph,node,"G",weight="weight")
            except nx.NetworkXNoPath: return dict(policy=policy,success=False,time=clock,changes=changes,trace=trace,reason="NO_PATH_CURRENT_MAP")
            changes+=int(replacement!=planned);planned=replacement
        if len(planned)<2 or not graph.has_edge(node,planned[1]):
            return dict(policy=policy,success=False,time=clock,changes=changes,trace=trace,reason="OLD_PLAN_INFEASIBLE")
        nxt=planned[1];duration=graph[node][nxt]["weight"]
        trace.append(dict(step=step,start=clock,origin=node,target=nxt,duration=duration,news=";".join(news)))
        clock+=duration;node=nxt;planned=planned[1:]
    return dict(policy=policy,success=False,time=clock,changes=changes,trace=trace,reason="EXECUTION_LIMIT")
results=[execute(p) for p in ["fixed","repair_blocked","rolling"]]
assert not results[0]["success"]
assert results[1]["success"] and results[2]["success"]


# %%
# ## 3. 按实际观测时刻复核执行轨迹
def verify_execution(result):
    graph=initial.copy();applied=set();clock=0;location="S"
    for row in result["trace"]:
        observe(graph,clock,applied)
        if row["origin"]!=location or row["start"]!=clock: return False
        if not graph.has_edge(location,row["target"]): return False
        if row["duration"]!=graph[location][row["target"]]["weight"]: return False
        clock+=row["duration"];location=row["target"]
    return clock==result["time"] and (not result["success"] or location=="G")
assert all(verify_execution(r) for r in results)
assert results[2]["time"]<results[1]["time"]
# 原计划失败耗时不是可执行方案的总成本，图中标成未完成。
summary=pd.DataFrame([{k:v for k,v in r.items() if k!="trace"} for r in results])


# %%
# ## 4. 给出触发规则和局限
summary.to_csv(OUT_DIR/"策略对照.csv",index=False,encoding="utf-8-sig")
pd.DataFrame([dict(policy=r["policy"],**row) for r in results for row in r["trace"]]).to_csv(OUT_DIR/"执行记录.csv",index=False,encoding="utf-8-sig")
successful=[r for r in results if r["success"]]
plt.bar([r["policy"] for r in successful],[r["time"] for r in successful]);plt.ylabel("完成任务的实际耗时")
plt.title("固定旧计划未完成，不作为有效低成本方案")
plt.tight_layout();plt.savefig(OUT_DIR/"重规划.png");plt.close()
best=min(successful,key=lambda r:r["time"])
save_results({"strategies":[{k:v for k,v in r.items() if k!="trace"} for r in results]},
             f"本情景推荐{best['policy']}，实际耗时{best['time']}、修改{best['changes']}次。"
             "仅在封路时修复会忽视仍可通行但变慢的道路；可将成本明显变化也设为事件触发。"
             "本例是图上反馈重规划，没有声称具备连续控制MPC的稳定性保证。")
