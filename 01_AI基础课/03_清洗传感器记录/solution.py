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
# ## 1. 保留原始输入
import pandas as pd, numpy as np, json
raw = pd.read_csv(DATA_DIR / 'sensor_dirty.csv')
work = raw.copy()
work['time'] = pd.to_datetime(work['time'], errors='coerce')
work['temperature'] = pd.to_numeric(work['temperature'], errors='coerce')
report = {'input_rows': len(raw), 'invalid_time': int(work['time'].isna().sum()), 'unparseable_or_missing': int(work['temperature'].isna().sum())}
print(work)


# %%
# ## 2. 定义行的身份与规则
work = work.dropna(subset=['time']).sort_values('time', kind='stable')
report['duplicate_rows'] = int(work.duplicated('time', keep='last').sum())
work = work.drop_duplicates('time', keep='last').copy()
work['out_of_range'] = ~work['temperature'].between(-40, 60) & work['temperature'].notna()
report['range_violations'] = int(work['out_of_range'].sum())
work.loc[work['out_of_range'], 'temperature'] = np.nan


# %%
# ## 3. 填充只使用过去：留下可追查标志
work['was_missing'] = work['temperature'].isna()
work['temperature_clean'] = work['temperature'].ffill(limit=1)
report['remaining_missing'] = int(work['temperature_clean'].isna().sum())
clean = work.dropna(subset=['temperature_clean']).copy()
assert clean['time'].is_unique
assert clean['temperature_clean'].between(-40, 60).all()
assert report['duplicate_rows'] == 1


# %%
# ## 4. 输出清洗日志，而不只交一张干净表
report['output_rows'] = len(clean)
clean.to_csv(OUT_DIR / 'clean.csv', index=False, encoding='utf-8-sig')
work.to_csv(OUT_DIR / 'audit.csv', index=False, encoding='utf-8-sig')
(OUT_DIR / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
plt.plot(work['time'], work['temperature'], 'o', label='有效原值')
plt.plot(work['time'], work['temperature_clean'], '--', label='单步前向填充')
plt.legend(); plt.ylabel('温度 / °C'); plt.xticks(rotation=25); plt.tight_layout()
plt.savefig(OUT_DIR / 'cleaning.png'); plt.close(); print(report)
