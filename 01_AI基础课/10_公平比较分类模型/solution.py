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
# ## 1. 先锁定独立测试集
import numpy as np, pandas as pd, json, time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
dataset = load_breast_cancer(as_frame=True)
X = dataset.data; y = (dataset.target == 0).astype(int)
X_dev,X_test,y_dev,y_test = train_test_split(X,y,test_size=0.25,stratify=y,random_state=10)
folds = list(StratifiedKFold(n_splits=3,shuffle=True,random_state=10).split(X_dev,y_dev))
assert set(X_dev.index).isdisjoint(X_test.index)


# %%
# ## 2. 预算意味着候选数与验证次数都一致
specs = {
 'logistic':(make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=10)),{'logisticregression__C':[0.1,1,10]}),
 'boosting':(HistGradientBoostingClassifier(max_iter=70,random_state=10),{'max_leaf_nodes':[3,7,15]})}
searches, comparison = {}, []
for name,(estimator,grid) in specs.items():
    start = time.perf_counter()
    search = GridSearchCV(estimator,grid,cv=folds,scoring='balanced_accuracy',n_jobs=1,refit=True)
    search.fit(X_dev,y_dev)
    searches[name] = search
    pd.DataFrame(search.cv_results_).to_csv(OUT_DIR/f'{name}_cv.csv',index=False)
    comparison.append({'model':name,'cv_score':float(search.best_score_),'seconds':time.perf_counter()-start,'best_parameters':search.best_params_})


# %%
# ## 3. 用交叉验证选模型，测试只作结算
selected = max(comparison,key=lambda row:row['cv_score'])['model']
result = pd.DataFrame({'actual':y_test.to_numpy()},index=X_test.index)
for row in comparison:
    name = row['model']; result[name] = searches[name].predict(X_test)
    row['test_balanced_accuracy'] = float(balanced_accuracy_score(y_test,result[name]))
metrics = {'selected_before_test':selected,'folds':3,'candidates_per_model':3,'comparison':comparison}
assert all(len(search.cv_results_['params']) == 3 for search in searches.values())


# %%
# ## 4. 保存全部候选，而非只保存最佳分数
result.to_csv(OUT_DIR/'test_predictions.csv',index=False)
(OUT_DIR/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
pd.DataFrame(comparison).set_index('model')[['cv_score','test_balanced_accuracy']].plot.bar(ylim=(0,1))
plt.ylabel('balanced accuracy'); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(OUT_DIR/'comparison.png'); plt.close(); print(metrics)
