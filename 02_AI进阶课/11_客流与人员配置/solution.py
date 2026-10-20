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
# ## 1. 定义目标与四份独立数据
import numpy as np,pandas as pd,json,math
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
rng=np.random.default_rng(111)
def generate_slots(n):
    x=pd.DataFrame({'hour':rng.integers(9,20,n),'weekend':rng.integers(0,2,n),'intensity':rng.uniform(0,1,n)})
    mean=35+45*np.exp(-((x['hour']-14)/3)**2)+20*x['weekend']+35*x['intensity']
    return x,rng.poisson(mean).astype(float)
X_train,y_train=generate_slots(500); X_cal,y_cal=generate_slots(200)
X_val,y_val=generate_slots(200); X_test,y_test=generate_slots(250)
capacity,wage,shortage_cost=20,30,8
model=HistGradientBoostingRegressor(max_iter=90,max_leaf_nodes=12,random_state=111).fit(X_train,y_train)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
linear_model=make_pipeline(StandardScaler(),Ridge(alpha=1)).fit(X_train,y_train)


# %%
# ## 2. 从点预测到校准区间
alpha=0.1
calibration_residuals=np.abs(y_cal-model.predict(X_cal))
rank=min(len(calibration_residuals),math.ceil((len(calibration_residuals)+1)*(1-alpha)))
q=float(np.sort(calibration_residuals)[rank-1])
def staffing_predictions(x):
    point=np.maximum(model.predict(x),0)
    return {'fixed':np.full(len(x),math.ceil(y_train.mean()/capacity),dtype=int),
            'linear_mean':np.ceil(np.maximum(linear_model.predict(x),0)/capacity).astype(int),
            'mean':np.ceil(point/capacity).astype(int),
            'buffered':np.ceil((point+q)/capacity).astype(int)}
def evaluate_staff(staff,actual):
    gap=np.maximum(actual-staff*capacity,0)
    return {'cost':float(np.mean(staff*wage+gap*shortage_cost)),'mean_staff':float(np.mean(staff)),'shortage_rate':float(np.mean(gap>0)),'mean_shortage':float(gap.mean())}


# %%
# ## 3. 用验证成本选择策略
validation_scores={name:evaluate_staff(staff,y_val) for name,staff in staffing_predictions(X_val).items()}
selected=min(validation_scores,key=lambda name:validation_scores[name]['cost'])
test_staff=staffing_predictions(X_test)
test_scores={name:evaluate_staff(staff,y_test) for name,staff in test_staff.items()}
point=model.predict(X_test); lower=np.maximum(point-q,0); upper=point+q
coverage=float(np.mean((y_test>=lower)&(y_test<=upper)))
model_comparison=[]
for name,estimator,policy in [('ridge',linear_model,'linear_mean'),('hist_gradient_boosting',model,'mean')]:
    model_comparison.append({'model':name,'validation_mae':float(mean_absolute_error(y_val,estimator.predict(X_val))),
                            'test_mae':float(mean_absolute_error(y_test,estimator.predict(X_test))),
                            'validation_decision_cost':validation_scores[policy]['cost'],'test_decision_cost':test_scores[policy]['cost']})
pd.DataFrame(model_comparison).to_csv(OUT_DIR/'prediction_vs_decision.csv',index=False)


# %%
# ## 4. 给出有条件的人员建议
results=X_test.assign(actual=y_test,point=point,lower=lower,upper=upper)
for name,staff in test_staff.items(): results[name+'_staff']=staff
results.to_csv(OUT_DIR/'staffing.csv',index=False)
metrics={'selected_on_validation':selected,'validation':validation_scores,'test':test_scores,'model_comparison':model_comparison,'test_mae':float(mean_absolute_error(y_test,point)),'interval_coverage':coverage,'mean_interval_width':float(np.mean(upper-lower)),'nominal_coverage':0.9}
(OUT_DIR/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
advice=f'验证集选择 {selected} 策略。独立测试平均成本 {test_scores[selected]["cost"]:.2f}，缺口时段比例 {test_scores[selected]["shortage_rate"]:.1%}；固定人数成本 {test_scores["fixed"]["cost"]:.2f}。经验区间覆盖 {coverage:.1%}。建议仅适用于每人容量 {capacity}、工资 {wage}、缺口代价 {shortage_cost} 的假设；容量、客流分布或工资改变后重新评估。'
if coverage < 0.9:
    advice += ' 本次有限测试集覆盖低于 90% 名义目标，须继续收集独立校准与验证数据并检查子群覆盖；不能把边际保证解读为当前每批数据必达标。'
(OUT_DIR/'决策建议.md').write_text('# 人员配置建议\n\n'+advice,encoding='utf-8')
pd.DataFrame(test_scores).T['cost'].plot.bar(); plt.ylabel('每时段平均总成本'); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(OUT_DIR/'staffing_cost.png'); plt.close(); print(metrics)
