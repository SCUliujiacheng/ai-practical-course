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
# ## 1. 固定笔记与同义改写问题
import json
import time
import numpy as np
started = time.perf_counter()
def finish(metrics):
    metrics["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
def barplot(labels, values, title, filename="comparison.png"):
    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename)
    plt.close(fig)

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import torch
torch.set_num_threads(2)
notes = [('D1', 'Cancel a laboratory booking at least four hours before it starts.'), ('D2', 'The library closes at ten in the evening on weekdays.'), ('D3', 'During heavy rain the outdoor event moves to the sports hall.'), ('D4', 'Report a broken bicycle and include its identification number.'), ('D5', 'Reimbursement requires an invoice and proof of payment.'), ('D6', 'Save random seeds and software versions before training a model.')]
questions = [('How do I withdraw my reservation?', 0), ('When must readers leave on Monday night?', 1), ('Where will people gather if it pours?', 2), ('What should I do about a damaged bike?', 3), ('Which receipts are needed to get my money back?', 4), ('What helps make an experiment repeatable?', 5)]
texts = [t for _, t in notes]
qs = [q for q, _ in questions]
gold = np.array([g for _, g in questions])
(DATA_DIR / 'semantic_notes.json').write_text(json.dumps({'notes': notes, 'questions': questions}, indent=2), encoding='utf-8')

# %%
# ## 2. 关键词基线与零匹配
v = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
lexical_matrix = v.fit_transform(texts)
t0 = time.perf_counter()
lexical_scores = (v.transform(qs) @ lexical_matrix.T).toarray()
lexical_seconds = time.perf_counter() - t0
lexical_top = lexical_scores.argmax(axis=1)
lexical_hit = (lexical_top == gold) & (lexical_scores.max(axis=1) > 0)

# %%
# ## 3. 本地向量编码与余弦相似度
model_path = COURSE_ROOT / 'assets/models/minilm'
if not (model_path / 'config.json').is_file():
    raise FileNotFoundError(f'缺少嵌入模型：{model_path}')
t0 = time.perf_counter()
encoder = SentenceTransformer(str(model_path), device='cpu', local_files_only=True)
load_seconds = time.perf_counter() - t0
t0 = time.perf_counter()
embeddings = encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
index_seconds = time.perf_counter() - t0
t0 = time.perf_counter()
query_embeddings = encoder.encode(qs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
semantic_scores = query_embeddings @ embeddings.T
query_seconds = time.perf_counter() - t0
semantic_top = semantic_scores.argmax(axis=1)
assert np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-05)

# %%
# ## 4. 比较效果与成本
rows = [{'query': q, 'gold': notes[g][0], 'lexical_top': notes[int(a)][0], 'lexical_max_score': float(lexical_scores[i].max()), 'semantic_top': notes[int(b)][0], 'semantic_max_score': float(semantic_scores[i].max())} for i, ((q, g), a, b) in enumerate(zip(questions, lexical_top, semantic_top))]
(OUT_DIR / 'comparison.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
scores = [float(lexical_hit.mean()), float(np.mean(semantic_top == gold))]
barplot(['关键词', 'MiniLM 向量'], scores, '同义改写检索 Recall@1')
finish({'lexical_recall_at_1': scores[0], 'semantic_recall_at_1': scores[1], 'model_load_seconds': round(load_seconds, 3), 'index_seconds': round(index_seconds, 3), 'semantic_query_seconds': round(query_seconds, 3), 'lexical_query_seconds': round(lexical_seconds, 3), 'language': 'English controlled comparison'})