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
# ## 1. 读入真实文本并处理重复
import pandas as pd, numpy as np, json
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
messages = pd.read_csv(COURSE_ROOT / 'assets/data/sms.tsv', sep='\t')
messages['text'] = messages['text'].fillna('').str.replace(r'\s+', ' ', regex=True).str.strip()
assert messages['label'].isin(['ham','spam']).all()
conflicts = messages.groupby('text')['label'].nunique()
messages = messages.loc[~messages['text'].isin(conflicts[conflicts > 1].index)]
messages = messages.loc[messages['text'].ne('')].drop_duplicates('text').reset_index(drop=True)
y = (messages['label'] == 'spam').astype(int)
train_idx, test_idx = train_test_split(np.arange(len(messages)), test_size=0.25, stratify=y, random_state=8)
assert set(messages.loc[train_idx,'text']).isdisjoint(set(messages.loc[test_idx,'text']))


# %%
# ## 2. 文本转换只学习训练集词表
model = make_pipeline(TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=2), MultinomialNB(alpha=0.5))
model.fit(messages.loc[train_idx,'text'], y.loc[train_idx])
print('词表大小', len(model.named_steps['tfidfvectorizer'].vocabulary_))
prediction = model.predict(messages.loc[test_idx,'text'])
probability = model.predict_proba(messages.loc[test_idx,'text'])[:,1]


# %%
# ## 3. 指标需要围绕误报与漏报
truth = y.loc[test_idx].to_numpy()
def classification_scores(truth, pred):
    return {'accuracy': float(np.mean(truth == pred)), 'precision': float(precision_score(truth,pred,zero_division=0)), 'recall': float(recall_score(truth,pred,zero_division=0)), 'f1': float(f1_score(truth,pred,zero_division=0))}
metrics = {'rows_after_deduplication': len(messages), 'baseline': classification_scores(truth,np.zeros_like(truth)), 'tfidf_nb': classification_scores(truth,prediction)}
matrix = confusion_matrix(truth, prediction, labels=[0,1])


# %%
# ## 4. 保存错误样例与矩阵
result = messages.loc[test_idx].copy(); result['prediction'] = prediction; result['spam_probability'] = probability
result.to_csv(OUT_DIR / 'predictions.csv', index=False, encoding='utf-8-sig')
result.loc[truth != prediction].to_csv(OUT_DIR / 'errors.csv', index=False, encoding='utf-8-sig')
(OUT_DIR / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
plt.imshow(matrix, cmap='Blues'); plt.xticks([0,1],['正常','垃圾']); plt.yticks([0,1],['正常','垃圾'])
for i in range(2):
    for j in range(2): plt.text(j,i,str(matrix[i,j]),ha='center',va='center')
plt.xlabel('预测'); plt.ylabel('真实'); plt.tight_layout(); plt.savefig(OUT_DIR / 'confusion.png'); plt.close(); print(metrics)
