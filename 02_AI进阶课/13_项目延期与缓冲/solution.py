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
# ## 1. 历史数据与依赖图
import numpy as np,pandas as pd,json
rng=np.random.default_rng(113)
names=['A','B','C','D','E','F']; means=np.array([2,5,7,3,2,6])
def new_projects(n,random):
    shared=random.normal(0,0.14,(n,1))
    individual=random.normal(0,0.12,(n,6))
    return means*np.exp(shared+individual-(0.14**2+0.12**2)/2)
history=new_projects(100,rng)
history_frame=pd.DataFrame(history,columns=names); history_frame.to_csv(DATA_DIR/'simulated_project_history.csv',index=False)
estimated_mean=history.mean(axis=0)
nominal_ready=estimated_mean[0]+max(estimated_mean[1],estimated_mean[2])
deadline=nominal_ready+estimated_mean[3]+estimated_mean[4]+2


# %%
# ## 2. 用预约机制明确缓冲的作用
def finish_times(durations,buffer_before):
    ready=durations[:,0]+np.maximum(durations[:,1],durations[:,2])
    slot=nominal_ready+buffer_before
    missed=ready>slot
    start_d=np.maximum(ready,slot)+3*missed
    finish_e=start_d+durations[:,3]+durations[:,4]
    finish_f=durations[:,0]+durations[:,5]
    return np.maximum(finish_e,finish_f),missed
tiny=np.array([[2,5,7,3,2,6]],dtype=float)
tiny_finish,_=finish_times(tiny,0)
assert tiny_finish[0]>=14


# %%
# ## 3. 用整行重抽样保留任务相关性
development=history[rng.integers(0,len(history),2000)]
policies={'end':0.0,'before':2.0}
development_results={}
for name,buffer_before in policies.items():
    finish,missed=finish_times(development,buffer_before)
    development_results[name]={'late_probability':float(np.mean(finish>deadline)),'mean_finish':float(finish.mean()),'missed_booking_rate':float(missed.mean())}
selected=min(development_results,key=lambda name:development_results[name]['late_probability'])


# %%
# ## 4. 独立未来情景验证并说明局限
future=new_projects(2000,np.random.default_rng(114))
future_results={}; rows=[]; finish_by_policy={}
for name,buffer_before in policies.items():
    finish,missed=finish_times(future,buffer_before); finish_by_policy[name]=finish
    future_results[name]={'late_probability':float(np.mean(finish>deadline)),'mean_finish':float(finish.mean()),'p90_finish':float(np.quantile(finish,0.9)),'missed_booking_rate':float(missed.mean())}
    rows.extend({'policy':name,'finish':float(f),'late':bool(f>deadline)} for f in finish)
metrics={'deadline':float(deadline),'selected_on_development':selected,'development':development_results,'independent_simulation':future_results,'simulation_only':True}
paired_difference=(finish_by_policy['before']>deadline).astype(float)-(finish_by_policy['end']>deadline).astype(float)
delta=float(paired_difference.mean()); monte_carlo_se=float(paired_difference.std(ddof=1)/np.sqrt(len(paired_difference)))
difference_interval=[delta-1.96*monte_carlo_se,delta+1.96*monte_carlo_se]
metrics['before_minus_end_late_probability']={'estimate':delta,'approx_95_monte_carlo_interval':difference_interval,'excludes_model_uncertainty':True}
pd.DataFrame(rows).to_csv(OUT_DIR/'finish_distribution.csv',index=False)
(OUT_DIR/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
advice=f'开发情景选择缓冲方案 {selected}。共同承诺工期 {deadline:.2f} 天；独立模拟延期率 {future_results[selected]["late_probability"]:.1%}，90% 完工分位 {future_results[selected]["p90_finish"]:.2f} 天。前置缓冲会推迟预约，但降低错过搭建队的风险；若不存在 3 天重约等待，本结论需重新计算。历史仅为模拟，不代表真实项目可靠性。'
advice += f' 前置减末端方案的延期率差为 {delta:.2%}，共同随机情景下的近似 95% 蒙特卡洛区间为 [{difference_interval[0]:.2%}, {difference_interval[1]:.2%}]；该区间不涵盖建模偏差。'
if difference_interval[0]<=0<=difference_interval[1]:
    advice += ' 目前差异不足以说明哪个方案稳定降低延期率；需结合预约失败后果、成本与更多真实记录决定。'
(OUT_DIR/'决策建议.md').write_text('# 工期建议\n\n'+advice,encoding='utf-8')
for name,finish in finish_by_policy.items(): plt.hist(finish,bins=35,alpha=0.5,label=name,density=True)
plt.axvline(deadline,color='black',ls='--',label='承诺日期'); plt.legend(); plt.xlabel('完工天数'); plt.ylabel('密度'); plt.tight_layout(); plt.savefig(OUT_DIR/'completion.png'); plt.close(); print(metrics)
