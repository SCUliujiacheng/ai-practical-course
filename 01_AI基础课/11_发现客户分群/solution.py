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
# ## 1. 定义行为特征而非臆测客户身份
import numpy as np,pandas as pd,json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
rng = np.random.default_rng(11)
centers = np.array([[2,60,40],[12,45,5],[4,260,18]])
parts = [rng.normal(center,[1,22,4],size=(90,3)) for center in centers]
customers = pd.DataFrame(np.maximum(np.vstack(parts),[0.1,5,0]),columns=['visits','basket','days_since'])
customers.to_csv(DATA_DIR/'simulated_customers.csv',index=False)
scaler = StandardScaler(); X_scaled = scaler.fit_transform(customers)


# %%
# ## 2. 尺度改变距离
cluster_models, scores = {}, []
for k in [2,3,4,5]:
    estimator = KMeans(n_clusters=k,n_init=10,random_state=11).fit(X_scaled)
    cluster_models[k] = estimator
    scores.append({'k':k,'silhouette':float(silhouette_score(X_scaled,estimator.labels_)),'inertia':float(estimator.inertia_)})
score_table = pd.DataFrame(scores)
selected_k = int(score_table.loc[score_table['silhouette'].idxmax(),'k'])
model = cluster_models[selected_k]


# %%
# ## 3. 返回原单位解释
customers['cluster'] = model.labels_
profiles = customers.groupby('cluster')[['visits','basket','days_since']].mean()
profiles['size'] = customers.groupby('cluster').size()
assert profiles['size'].sum() == len(customers)
print(profiles)


# %%
# ## 4. 保存选择依据与分群
customers.to_csv(OUT_DIR/'assignments.csv',index=False)
profiles.to_csv(OUT_DIR/'profiles.csv'); score_table.to_csv(OUT_DIR/'k_comparison.csv',index=False)
metrics = {'selected_k':selected_k,'silhouette':float(score_table.set_index('k').loc[selected_k,'silhouette']),'cluster_sizes':profiles['size'].to_dict()}
(OUT_DIR/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
plt.scatter(customers['visits'],customers['basket'],c=customers['cluster'],cmap='tab10',s=20)
plt.xlabel('购买次数'); plt.ylabel('客单价 / 教学单位'); plt.tight_layout(); plt.savefig(OUT_DIR/'clusters.png'); plt.close()
