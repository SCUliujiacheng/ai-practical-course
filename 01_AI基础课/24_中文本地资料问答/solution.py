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
# ## 1. 手册、验证查询与独立测试题
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
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(2)
docs = [('D1', '开放日签到地点是一号楼大厅。'), ('D2', '志愿者集合时间是早上8点。'), ('D3', '遇到大雨，活动转入体育馆。'), ('D4', '报销材料必须包含发票和支付凭证。')]
validation = [('开放日在哪里签到', 0), ('志愿者几点集合', 1), ('今晚股票会涨吗', None), ('校长的生日是哪天', None)]
tests = [('签到地点在哪里', 0, '一号楼'), ('下大雨活动转到哪里', 2, '体育馆'), ('明天股票价格是多少', None, None), ('校长的生日是什么时候', None, None)]
v = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
index = v.fit_transform([t for _, t in docs])

def find_evidence(question):
    scores = (index @ v.transform([question]).T).toarray().ravel()
    i = int(scores.argmax())
    return (i, float(scores[i]))
validation_scores = [find_evidence(q)[1] for q, _ in validation]
candidates = np.arange(0.1, 0.81, 0.05)
threshold = float(min(candidates, key=lambda t: sum(((s >= t) != (g is not None) for s, (_, g) in zip(validation_scores, validation)))))
print('验证集选定阈值', threshold)

# %%
# ## 2. 因果语言模型与提示模板
model_path = COURSE_ROOT / 'assets/models/qwen3-0.6b'
if not (model_path / 'config.json').is_file():
    raise FileNotFoundError(f'缺少本地生成模型：{model_path}')
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float32)
model.eval()

def generate_answer(question, doc_index):
    doc_id, text = docs[doc_index]
    messages = [{'role': 'system', 'content': '你是资料助手。只依据资料回答，用一句简短中文，不补充其他事实。'}, {'role': 'user', 'content': f'资料[{doc_id}]：{text} 问题：{question} 直接给出答案。'}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors='pt')
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=32, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

# %%
# ## 3. 门控、生成和支持检查
rows = []
for question, gold, expected in tests:
    i, score = find_evidence(question)
    abstained = score < threshold
    t0 = time.perf_counter()
    raw = '资料不足，无法回答。' if abstained else generate_answer(question, i)
    supported = bool(not abstained and gold == i and (expected is not None) and (expected in raw) and (expected in docs[i][1]))
    rendered = raw if abstained else raw + f' [来源:{docs[i][0]}]'
    rows.append({'question': question, 'gold_doc': None if gold is None else docs[gold][0], 'retrieved_doc': docs[i][0], 'retrieval_score': score, 'abstained': abstained, 'raw_generation': raw, 'answer_with_source': rendered, 'expected_phrase': expected, 'phrase_support_proxy': supported, 'seconds': round(time.perf_counter() - t0, 3)})
    print(question, rendered)

# %%
# ## 4. 结果拆开评价
answerable = [r for r in rows if r['gold_doc'] is not None]
unanswerable = [r for r in rows if r['gold_doc'] is None]
recall = float(np.mean([r['retrieved_doc'] == r['gold_doc'] for r in answerable]))
support = float(np.mean([r['phrase_support_proxy'] for r in answerable]))
refusal = float(np.mean([r['abstained'] for r in unanswerable]))
false_refusal = float(np.mean([r['abstained'] for r in answerable]))
(OUT_DIR / 'answers_for_review.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
(DATA_DIR / 'handbook.json').write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding='utf-8')
barplot(['来源命中', '短语支持代理', '无答案拒答', '错误拒答'], [recall, support, refusal, false_refusal], 'RAG 分项评价')
finish({'answerable_retrieval_recall': recall, 'phrase_support_proxy_rate': support, 'unanswerable_refusal_rate': refusal, 'false_refusal_rate': false_refusal, 'threshold': threshold, 'test_questions': len(tests), 'human_full_claim_review': 'required; phrase proxy is not full verification', 'model': 'Qwen/Qwen3-0.6B'})