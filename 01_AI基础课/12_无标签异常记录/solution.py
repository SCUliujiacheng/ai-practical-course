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
# ## 1. 历史参考与未来复核集分开
import numpy as np,pandas as pd,json
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score,recall_score
rng = np.random.default_rng(12)
train = rng.normal([40,2,10],[3,0.3,1],size=(450,3))
normal = rng.normal([40,2,10],[3,0.3,1],size=(180,3))
abnormal = rng.normal([43,3.1,13],[4,0.4,1],size=(20,3))
test = np.vstack([normal,abnormal]); truth = np.r_[np.zeros(180,dtype=int),np.ones(20,dtype=int)]
order = rng.permutation(len(test)); test,truth = test[order],truth[order]
pd.DataFrame(train,columns=['temperature','vibration','current']).to_csv(DATA_DIR/'reference_history.csv',index=False)


# %%
# ## 2. 简单基线：最大绝对标准分数
mean,scale = train.mean(axis=0),train.std(axis=0)
z_train = np.max(np.abs((train-mean)/scale),axis=1)
z_test = np.max(np.abs((test-mean)/scale),axis=1)
z_threshold = float(np.quantile(z_train,0.95))
z_alert = z_test > z_threshold


# %%
# ## 3. Isolation Forest：以隔离难度产生分数
model = IsolationForest(n_estimators=100,random_state=12,n_jobs=1).fit(train)
train_scores = -model.score_samples(train)
test_scores = -model.score_samples(test)
threshold = float(np.quantile(train_scores,0.95))
alert = test_scores > threshold
assert len(test_scores) == len(test)


# %%
# ## 4. 输出排序后的复核队列
def score_alert(pred):
    return {'alerts':int(pred.sum()),'precision':float(precision_score(truth,pred,zero_division=0)),'recall':float(recall_score(truth,pred,zero_division=0))}
metrics = {'zscore':score_alert(z_alert),'isolation_forest':score_alert(alert),'threshold':threshold}
result = pd.DataFrame(test,columns=['temperature','vibration','current']).assign(score=test_scores,alert=alert,simulated_truth=truth)
result.sort_values('score',ascending=False).to_csv(OUT_DIR/'review_queue.csv',index=False)
(OUT_DIR/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
plt.scatter(test[:,1],test[:,2],c=test_scores,cmap='magma'); plt.colorbar(label='异常分数'); plt.xlabel('振动'); plt.ylabel('电流'); plt.tight_layout(); plt.savefig(OUT_DIR/'anomalies.png'); plt.close(); print(metrics)
