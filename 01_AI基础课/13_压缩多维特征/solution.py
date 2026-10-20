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
# ## 1. 理解矩阵形状
import numpy as np,pandas as pd,json
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
dataset = load_wine(as_frame=True)
train_idx,test_idx = train_test_split(np.arange(len(dataset.data)),test_size=0.25,random_state=13,stratify=dataset.target)
scaler = StandardScaler().fit(dataset.data.iloc[train_idx])
train = scaler.transform(dataset.data.iloc[train_idx]); test = scaler.transform(dataset.data.iloc[test_idx])
assert train.shape[1] == 13
print('训练矩阵',train.shape,'测试矩阵',test.shape)


# %%
# ## 2. 用训练集方差决定保留维数
full = PCA().fit(train)
cumulative = np.cumsum(full.explained_variance_ratio_)
k = int(np.searchsorted(cumulative,0.8)+1)
pca = PCA(n_components=k).fit(train)
compressed = pca.transform(test)
assert compressed.shape == (len(test),k)


# %%
# ## 3. 逆变换检查丢失的信息
reconstructed = pca.inverse_transform(compressed)
baseline = np.tile(train.mean(axis=0),(len(test),1))
reconstruction_mse = float(np.mean((test-reconstructed)**2))
baseline_mse = float(np.mean((test-baseline)**2))
comparison = pd.DataFrame({'feature':dataset.feature_names,'standardized_mse':np.mean((test-reconstructed)**2,axis=0)})
assert reconstruction_mse <= baseline_mse + 1e-10


# %%
# ## 4. 保存方差曲线与二维投影
metrics = {'n_features':13,'retained_components':k,'training_variance_retained':float(cumulative[k-1]),'test_reconstruction_mse':reconstruction_mse,'mean_baseline_mse':baseline_mse}
comparison.to_csv(OUT_DIR/'feature_errors.csv',index=False)
pd.DataFrame(compressed).to_csv(OUT_DIR/'compressed_test.csv',index=False)
(OUT_DIR/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
fig,axes = plt.subplots(1,2,figsize=(10,4)); axes[0].plot(np.arange(1,14),cumulative,'o-'); axes[0].axhline(0.8,ls='--'); axes[0].set(xlabel='维数',ylabel='累计解释方差')
projection = full.transform(test); axes[1].scatter(projection[:,0],projection[:,1],c=dataset.target.iloc[test_idx],cmap='tab10'); axes[1].set(xlabel='PC1',ylabel='PC2')
fig.tight_layout(); fig.savefig(OUT_DIR/'pca.png'); plt.close(fig); print(metrics)
