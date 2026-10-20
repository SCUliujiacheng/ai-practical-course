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
# ## 1. 定义任务与规则对照
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

import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
torch.set_num_threads(2)
examples = [('I love this wonderful book.', 1), ('The service was terrible and slow.', 0), ('It is not good at all.', 0), ('A reliable and useful product.', 1), ('I regret buying this item.', 0), ('The staff were friendly and helpful.', 1)]

def lexicon_predict(text):
    tokens = set(re.findall('[a-z]+', text.lower()))
    return int(len(tokens & {'love', 'wonderful', 'good', 'reliable', 'useful', 'friendly', 'helpful'}) >= len(tokens & {'terrible', 'slow', 'regret', 'bad', 'hate'}))
baseline_pred = [lexicon_predict(t) for t, _ in examples]

# %%
# ## 2. 本地加载与批分词
model_path = COURSE_ROOT / 'assets/models/flan-t5-small'
if not (model_path / 'config.json').is_file():
    raise FileNotFoundError(f'缺少本地模型：{model_path}')
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
model.eval()
prompts = ['Classify the sentiment as positive or negative. Answer with one word. Text: ' + text for text, _ in examples]
batch = tokenizer(prompts, padding=True, truncation=True, max_length=128, return_tensors='pt')
assert batch['input_ids'].shape == batch['attention_mask'].shape
assert set(batch['attention_mask'].unique().tolist()) <= {0, 1}
print('输入形状', batch['input_ids'].shape)

# %%
# ## 3. 生成与独立解析
t0 = time.perf_counter()
with torch.inference_mode():
    generated = model.generate(**batch, max_new_tokens=8, do_sample=False, num_beams=1)
raw_outputs = tokenizer.batch_decode(generated, skip_special_tokens=True)
generation_seconds = time.perf_counter() - t0

def parse_label(text):
    return {'positive': 1, 'negative': 0}.get(text.strip().lower().strip('.! '), -1)
model_pred = [parse_label(t) for t in raw_outputs]

# %%
# ## 4. 保留真实输出与错误
truth = np.array([label for _, label in examples])
rows = [{'text': t, 'expected': y, 'raw': raw, 'parsed': int(p), 'baseline': int(b)} for (t, y), raw, p, b in zip(examples, raw_outputs, model_pred, baseline_pred)]
(OUT_DIR / 'predictions.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
scores = [float(np.mean(truth == baseline_pred)), float(np.mean(truth == model_pred))]
barplot(['词典规则', 'FLAN-T5'], scores, '本地 Transformer 与基线')
finish({'baseline_accuracy': scores[0], 'transformer_accuracy': scores[1], 'format_valid_rate': float(np.mean(np.array(model_pred) != -1)), 'generation_seconds': round(generation_seconds, 3), 'examples': len(examples), 'model': 'google/flan-t5-small'})