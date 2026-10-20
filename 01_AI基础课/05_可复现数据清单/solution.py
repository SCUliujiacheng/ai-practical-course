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
# ## 1. 文件和标签是两种信息
import json, hashlib
import pandas as pd
from sklearn.model_selection import train_test_split
sample_dir = DATA_DIR / 'samples'; sample_dir.mkdir(exist_ok=True)
rows = []
for person in range(30):
    label = person % 2
    for visit in range(2):
        sample_id = f'p{person:02d}_v{visit}'
        path = sample_dir / (sample_id + '.txt')
        path.write_text(f'Teaching record for person {person}, visit {visit}.', encoding='utf-8')
        rows.append({'sample_id': sample_id, 'person_id': person, 'label': label, 'path': path.relative_to(CASE_DIR).as_posix()})
manifest = pd.DataFrame(rows)
assert manifest['sample_id'].is_unique


# %%
# ## 2. 在独立单位上分层抽样
people = manifest[['person_id', 'label']].drop_duplicates()
train_people, held_people = train_test_split(people, test_size=0.4, stratify=people['label'], random_state=5)
val_people, test_people = train_test_split(held_people, test_size=0.5, stratify=held_people['label'], random_state=5)
split_map = {}
for split, subset in [('train', train_people), ('validation', val_people), ('test', test_people)]:
    split_map.update({int(person): split for person in subset['person_id']})
manifest['split'] = manifest['person_id'].map(split_map)


# %%
# ## 3. 自动检查泄漏与内容版本
groups = {name: set(part['person_id']) for name, part in manifest.groupby('split')}
assert groups['train'].isdisjoint(groups['test'])
assert groups['train'].isdisjoint(groups['validation'])
assert groups['test'].isdisjoint(groups['validation'])
manifest['sha256'] = [hashlib.sha256((CASE_DIR / p).read_bytes()).hexdigest() for p in manifest['path']]
assert manifest['sha256'].str.len().eq(64).all()


# %%
# ## 4. 清单成为数据集入口
manifest.to_json(OUT_DIR / 'manifest.jsonl', orient='records', lines=True, force_ascii=False)
manifest.to_csv(OUT_DIR / 'manifest.csv', index=False)
counts = pd.crosstab(manifest['split'], manifest['label'])
metrics = {'samples': len(manifest), 'people': len(people), 'counts': manifest['split'].value_counts().to_dict(), 'group_leakage': False}
(OUT_DIR / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
counts.plot.bar(); plt.ylabel('样本数'); plt.xticks(rotation=0); plt.tight_layout()
plt.savefig(OUT_DIR / 'splits.png'); plt.close(); print(counts)
