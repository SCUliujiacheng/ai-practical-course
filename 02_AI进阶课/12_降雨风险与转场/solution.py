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
# ## 1. 列出天气状态与损失表
import numpy as np,pandas as pd,json
prior=0.35; rain_loss=12000; early_indoor=3000; late_indoor=3800; wait_fee=500
sensitivity=0.8; false_positive=0.2
loss_table=pd.DataFrame({'no_rain':[0,early_indoor],'rain':[rain_loss,early_indoor]},index=['outdoor','indoor_now'])
print(loss_table)


# %%
# ## 2. 等待带来信息也带来代价
def signal_branches(p,sensitivity=0.8,false_positive=0.2):
    wet_probability=p*sensitivity+(1-p)*false_positive
    branches=[]
    for label,prob, rain_joint in [('wet',wet_probability,p*sensitivity),('dry',1-wet_probability,p*(1-sensitivity))]:
        posterior=rain_joint/prob if prob>0 else 0.0
        branches.append({'signal':label,'probability':prob,'posterior_rain':posterior})
    return branches
branches=signal_branches(prior,sensitivity,false_positive)
assert np.isclose(sum(b['probability']*b['posterior_rain'] for b in branches),prior)


# %%
# ## 3. 从决策树末端往前计算
def expected_losses(p,waiting_fee=wait_fee):
    branch_rows=signal_branches(p,sensitivity,false_positive)
    expected_wait=waiting_fee
    for branch in branch_rows:
        outdoor=branch['posterior_rain']*rain_loss
        branch['action']='outdoor' if outdoor<=late_indoor else 'indoor_later'
        branch['loss_after_signal']=min(outdoor,late_indoor)
        expected_wait+=branch['probability']*branch['loss_after_signal']
    return {'outdoor':p*rain_loss,'indoor_now':early_indoor,'wait':expected_wait},branch_rows
losses,branch_decisions=expected_losses(prior)
selected=min(losses,key=losses.get)
curve=pd.DataFrame([{'rain_probability':p,**expected_losses(p)[0]} for p in np.linspace(0,1,101)])


# %%
# ## 4. 交付规则与改变建议的条件
metrics={'assumed_rain_probability':prior,'expected_losses':losses,'selected':selected,'immediate_indoor_threshold':early_indoor/rain_loss,'signal_sensitivity':sensitivity,'signal_false_positive':false_positive}
curve.to_csv(OUT_DIR/'decision_curve.csv',index=False)
pd.DataFrame(branch_decisions).to_csv(OUT_DIR/'signal_actions.csv',index=False)
(OUT_DIR/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
advice=f'在假设降雨概率 {prior:.0%} 下，选择 {selected}，预期损失 {losses[selected]:.2f}；户外 {losses["outdoor"]:.2f}，立即室内 {losses["indoor_now"]:.2f}，等待 {losses["wait"]:.2f}。等待信号灵敏度 {sensitivity:.0%}、误报率 {false_positive:.0%}，需在实际取得新信息后按 signal_actions.csv 执行。所有数值均为教学情景，若场地或成本变化须重算。'
(OUT_DIR/'决策建议.md').write_text('# 转场建议\n\n'+advice,encoding='utf-8')
curve.set_index('rain_probability')[['outdoor','indoor_now','wait']].plot(); plt.xlabel('假设降雨概率'); plt.ylabel('预期损失'); plt.tight_layout(); plt.savefig(OUT_DIR/'decision.png'); plt.close(); print(metrics)
