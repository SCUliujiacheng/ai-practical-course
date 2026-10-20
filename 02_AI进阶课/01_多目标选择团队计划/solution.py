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
# ## 1. 把文字要求变成表格
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

plans = pd.DataFrame({"plan": ["快速", "均衡", "节省"],
                      "days": [7, 9, 11], "cost": [1200, 900, 700],
                      "max_hours": [36, 27, 24]})
limits = {"days": 12, "cost": 1250, "max_hours": 40}
columns = list(limits)
def feasible_mask(frame, caps):
    return np.array([all(row[key] <= cap for key, cap in caps.items())
                     for _, row in frame.iterrows()])
candidates = plans.loc[feasible_mask(plans, limits)].copy()
assert not candidates.empty
plans.to_csv(DATA_DIR / "计划.csv", index=False, encoding="utf-8-sig")
print(candidates)


# %%
# ## 2. 先检查支配关系
def pareto_mask(values):
    values = np.asarray(values, dtype=float)
    return np.array([not any(np.all(values[j] <= values[i]) and
                             np.any(values[j] < values[i])
                             for j in range(len(values)) if j != i)
                     for i in range(len(values))])
candidates["pareto"] = pareto_mask(candidates[columns].to_numpy())
assert pareto_mask([[1, 2], [2, 3]]).tolist() == [True, False]
print(candidates[["plan", "pareto"]])


# %%
# ## 3. 用明确权重比较，再做敏感性分析
values = candidates[columns].to_numpy(float)
span = np.ptp(values, axis=0)
normalized = (values - values.min(axis=0)) / np.where(span == 0, 1, span)
def choose(weights):
    weights = np.asarray(weights, dtype=float)
    assert len(weights) == len(columns) and np.all(weights >= 0)
    assert np.isclose(weights.sum(), 1)
    scores = normalized @ weights
    return int(np.argmin(scores)), scores
selected, scores = choose([0.5, 0.3, 0.2])
candidates["score"] = scores
sensitivity = []
for urgency in np.linspace(0, 1, 21):
    idx, _ = choose([urgency, 0.6 * (1 - urgency), 0.4 * (1 - urgency)])
    sensitivity.append({"urgency_weight": urgency,
                        "chosen": candidates.iloc[idx]["plan"]})
sensitivity = pd.DataFrame(sensitivity)
assert feasible_mask(candidates, limits).all()


# %%
# ## 4. 输出可复核的建议
candidates.to_csv(OUT_DIR / "方案对照.csv", index=False, encoding="utf-8-sig")
sensitivity.to_csv(OUT_DIR / "权重敏感性.csv", index=False, encoding="utf-8-sig")
ax = candidates.set_index("plan")["score"].plot.bar(color=["#467a9e"] * len(candidates))
ax.set_ylabel("归一化加权代价（越小越好）")
ax.set_title("权重：工期0.5、成本0.3、负担0.2")
plt.tight_layout(); plt.savefig(OUT_DIR / "方案比较.png"); plt.close()
chosen = candidates.iloc[selected]
save_results({"chosen": chosen["plan"], "score": float(chosen["score"]),
              "feasible_count": len(candidates),
              "pareto_count": int(candidates["pareto"].sum())},
             f"在工期/成本/负担权重为0.5/0.3/0.2时，推荐{chosen['plan']}方案，"
             f"工期{chosen['days']}天、成本{chosen['cost']}、最大负担{chosen['max_hours']}小时。"
             f"当前有{int(candidates['pareto'].sum())}个非支配方案；权重或预算变化时重新比较。")
