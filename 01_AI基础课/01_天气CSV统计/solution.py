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
# ## 1. 读入：文件中的数字最初也是字符串
import csv, json
with (DATA_DIR / 'weather.csv').open(encoding='utf-8', newline='') as stream:
    raw_rows = list(csv.DictReader(stream))
print(raw_rows[:2], type(raw_rows[0]['temp_max']))
assert len(raw_rows) == 10


# %%
# ## 2. 转换：建立清楚的数据约定
from datetime import date
rows = []
for raw in raw_rows:
    value = float(raw['temp_max'])
    date.fromisoformat(raw['date'])
    assert -60 <= value <= 60
    rows.append({'date': raw['date'], 'temp_max': value})
print(rows[0])


# %%
# ## 3. 函数：把统计过程变成能重复调用的步骤
def summarize(records, threshold=30):
    if not records:
        raise ValueError('没有记录，不能计算平均值')
    hottest = records[0]
    total = 0.0
    count = 0
    for row in records:
        total += row['temp_max']
        if row['temp_max'] > hottest['temp_max']:
            hottest = row
        if row['temp_max'] >= threshold:
            count += 1
    return {'hottest_date': hottest['date'], 'hottest_temp': hottest['temp_max'],
            'mean_temp': total / len(records), 'hot_days': count, 'threshold': threshold}
metrics = summarize(rows)
assert metrics['hottest_date'] == '2024-07-06'
assert metrics['hot_days'] == 4
print(metrics)


# %%
# ## 4. 交付：数值与图互相检查
(OUT_DIR / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
plt.plot([r['date'][5:] for r in rows], [r['temp_max'] for r in rows], marker='o')
plt.axhline(30, linestyle='--', color='orange', label='30°C')
plt.ylabel('最高温度 / °C'); plt.xlabel('日期'); plt.legend(); plt.tight_layout()
plt.savefig(OUT_DIR / 'temperature.png'); plt.close()
