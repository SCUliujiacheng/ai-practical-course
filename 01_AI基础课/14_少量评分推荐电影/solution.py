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
# ## 1. 评分是部分观察，不是完整偏好
import numpy as np,pandas as pd,json
from sklearn.metrics.pairwise import cosine_similarity
rng = np.random.default_rng(14); n_users,n_items = 60,15
records=[]
for user in range(n_users):
    taste = user%3
    chosen = rng.choice(n_items,8,replace=False)
    for event,item in enumerate(chosen):
        rating = np.clip(2.8 + 1.8*(item%3==taste) + rng.normal(0,0.45),1,5)
        records.append((user,int(item),event,float(rating)))
ratings = pd.DataFrame(records,columns=['user','item','event','rating'])
train = ratings.loc[ratings['event']<6].copy(); test = ratings.loc[ratings['event']>=6].copy()
matrix = train.pivot(index='user',columns='item',values='rating').reindex(index=range(n_users),columns=range(n_items))
ratings.to_csv(DATA_DIR/'simulated_ratings.csv',index=False)


# %%
# ## 2. 从训练评分学习电影相似度
user_means = matrix.mean(axis=1)
residual = matrix.sub(user_means,axis=0).fillna(0)
similarity = np.maximum(cosine_similarity(residual.T),0)
np.fill_diagonal(similarity,0)
popularity = matrix.mean(axis=0).fillna(train['rating'].mean()).to_numpy()


# %%
# ## 3. 为未看过的电影排序
def recommend(user, k=3, collaborative=True):
    if user not in matrix.index:
        return np.argsort(-popularity)[:k]
    seen = matrix.loc[user].notna().to_numpy()
    if collaborative:
        numerator = similarity @ residual.loc[user].to_numpy()
        denominator = similarity @ seen.astype(float)
        scores = float(user_means.loc[user]) + np.divide(numerator,denominator,out=np.zeros(n_items),where=denominator>0)
    else:
        scores = popularity.copy()
    scores[seen] = -np.inf
    return np.argsort(-scores)[:k]
assert not set(recommend(0)).intersection(set(train.loc[train['user']==0,'item']))


# %%
# ## 4. 留出正反馈命中率
evaluation=[]
for user,part in test.groupby('user'):
    positives=set(part.loc[part['rating']>=4,'item'])
    if not positives: continue
    cf=recommend(user); popular=recommend(user,collaborative=False)
    evaluation.append({'user':int(user),'cf_hit':int(bool(set(cf)&positives)),'popular_hit':int(bool(set(popular)&positives)),'recommendations':','.join(map(str,cf))})
evaluation=pd.DataFrame(evaluation)
metrics={'eligible_users':len(evaluation),'cf_hit_at_3':float(evaluation['cf_hit'].mean()),'popular_hit_at_3':float(evaluation['popular_hit'].mean())}
evaluation.to_csv(OUT_DIR/'recommendations.csv',index=False)
(OUT_DIR/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
plt.imshow(similarity,cmap='Blues'); plt.colorbar(label='电影相似度'); plt.xlabel('电影编号'); plt.ylabel('电影编号'); plt.tight_layout(); plt.savefig(OUT_DIR/'item_similarity.png'); plt.close(); print(metrics)
