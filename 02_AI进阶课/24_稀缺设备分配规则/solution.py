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
# ## 1. 把价值、出价和到达顺序分开
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

values=np.array([9.,6.,4.,2.]);arrival=[2,1,3,0]
def second_price(bids):
    bids=np.asarray(bids,float)
    if len(bids)<2 or np.any(bids<0):raise ValueError("至少两人且出价非负")
    order=sorted(range(len(bids)),key=lambda i:(-bids[i],i))
    winner=order[0];payment=float(bids[order[1]])
    return winner,payment
def utility(values,winner,payment):
    rewards=np.zeros(len(values));rewards[winner]=values[winner]-payment
    return rewards
pd.DataFrame(dict(team=range(4),value=values,arrival_rank=[arrival.index(i) for i in range(4)])).to_csv(DATA_DIR/"团队价值.csv",index=False)


# %%
# ## 2. 实现三个可检查机制
first_winner=arrival[0]
rng=np.random.default_rng(24);draws=rng.integers(0,len(values),size=4000)
auction_winner,payment=second_price(values)
counts=np.bincount(draws,minlength=len(values))
expected_lottery_value=float(values.mean())
assert auction_winner==0 and payment==6
assert np.isclose(utility(values,auction_winner,payment).sum()+payment,values[auction_winner])
table=pd.DataFrame([dict(rule="先到先得",allocated_value=values[first_winner],payment=0),
                    dict(rule="均匀抽签期望",allocated_value=expected_lottery_value,payment=0),
                    dict(rule="第二价格拍卖",allocated_value=values[auction_winner],payment=payment)])


# %%
# ## 3. 单方改变报价的反例检查
deviations=[]
for person,value in enumerate(values):
    truthful_winner,truthful_pay=second_price(values)
    truthful_u=utility(values,truthful_winner,truthful_pay)[person]
    for bid in np.arange(0,13,.5):
        bids=values.copy();bids[person]=bid
        winner,pay=second_price(bids);u=utility(values,winner,pay)[person]
        deviations.append(dict(person=person,bid=bid,utility=float(u),truthful_utility=float(truthful_u)))
        assert u<=truthful_u+1e-9
# 获得设备的人支付第二高报价；卖方收入不等于被创造的社会价值。
assert payment<=values[auction_winner]


# %%
# ## 4. 同时报告效率与分配机会
table.to_csv(OUT_DIR/"规则对照.csv",index=False,encoding="utf-8-sig")
pd.DataFrame(deviations).to_csv(OUT_DIR/"报价偏离检查.csv",index=False)
pd.DataFrame(dict(team=range(4),draw_count=counts,draw_frequency=counts/len(draws))).to_csv(OUT_DIR/"抽签频率.csv",index=False)
table.set_index("rule")["allocated_value"].plot.bar(color="#688cb0")
plt.ylabel("获配者价值／抽签期望价值");plt.xticks(rotation=15)
plt.tight_layout();plt.savefig(OUT_DIR/"分配规则.png");plt.close()
save_results({"first_winner":first_winner,"first_value":float(values[first_winner]),"lottery_expected_value":expected_lottery_value,
              "lottery_observed_value":float(values[draws].mean()),"auction_winner":auction_winner,
              "auction_payment":payment,"deviation_checks":len(deviations)},
             f"价值效率优先且满足私人价值等假设时，第二价格规则将设备分配给团队{auction_winner}，支付{payment:g}，价值{values[auction_winner]:g}。"
             f"均匀抽签期望价值{expected_lottery_value:g}但机会相等。公平或预算目标不同，应明确另选规则。")
