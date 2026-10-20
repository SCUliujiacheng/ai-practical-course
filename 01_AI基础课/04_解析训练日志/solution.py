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
# ## 1. 读取混杂日志
import re, json
import pandas as pd
lines = (DATA_DIR / 'training.log').read_text(encoding='utf-8').splitlines()
print('\n'.join(lines[:4]))


# %%
# ## 2. 正则表达式提取有名字的字段
number = r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?'
pattern = re.compile(r'epoch=(?P<epoch>\d+)\s+train=(?P<train>'+number+r')\s+val=(?P<val>'+number+r')')
parsed, skipped = [], []
for line_no, line in enumerate(lines, 1):
    match = pattern.search(line)
    if match:
        parsed.append({'epoch': int(match['epoch']), 'train': float(match['train']), 'val': float(match['val'])})
    else:
        skipped.append({'line': line_no, 'text': line})
table = pd.DataFrame(parsed).sort_values('epoch')
assert table['epoch'].is_unique
assert (table[['train', 'val']] >= 0).all().all()


# %%
# ## 3. 验证集选择停止点
best = table.loc[table['val'].idxmin()]
table['gap'] = table['val'] - table['train']
metrics = {'parsed_epochs': len(table), 'skipped_lines': len(skipped), 'best_epoch': int(best['epoch']), 'best_validation_loss': float(best['val'])}
assert metrics['best_epoch'] == 4
print(table, metrics)


# %%
# ## 4. 保存供复核的结构化结果
table.to_csv(OUT_DIR / 'epochs.csv', index=False)
(OUT_DIR / 'unparsed.json').write_text(json.dumps(skipped, ensure_ascii=False, indent=2), encoding='utf-8')
(OUT_DIR / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
plt.plot(table['epoch'], table['train'], label='训练损失')
plt.plot(table['epoch'], table['val'], label='验证损失')
plt.axvline(best['epoch'], linestyle='--', color='grey'); plt.legend(); plt.xlabel('epoch'); plt.ylabel('损失')
plt.tight_layout(); plt.savefig(OUT_DIR / 'learning_curve.png'); plt.close()
