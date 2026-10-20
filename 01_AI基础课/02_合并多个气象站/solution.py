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
# ## 1. 多文件读取：每个文件是一块同结构数据
import pandas as pd, json
frames = []
for path in sorted(DATA_DIR.glob('station_*.csv')):
    frame = pd.read_csv(path, parse_dates=['date'])
    frame['source'] = path.name
    frames.append(frame)
weather = pd.concat(frames, ignore_index=True)
assert len(weather) == 9
assert not weather.duplicated(['station', 'date']).any()
print(weather.head())


# %%
# ## 2. 属性连接：站点表必须一站一行
stations = pd.read_csv(DATA_DIR / 'stations.csv')
merged = weather.merge(stations, on='station', how='left', validate='many_to_one')
assert len(merged) == len(weather)
assert merged['city'].notna().all()
print(merged[['station', 'city', 'date', 'temp_max']])


# %%
# ## 3. 分组：一行代表什么会改变平均值
city_day = merged.groupby(['city', 'date'], as_index=False)['temp_max'].mean()
summary = city_day.groupby('city')['temp_max'].agg(['mean', 'max', 'count'])
daily_wide = city_day.pivot(index='date', columns='city', values='temp_max')
print(summary)
assert len(city_day) == 6


# %%
# ## 4. 导出与检查：让另一人能复算
merged.to_csv(OUT_DIR / 'merged.csv', index=False, encoding='utf-8-sig')
city_day.to_csv(OUT_DIR / 'city_daily.csv', index=False, encoding='utf-8-sig')
metrics = {'records': len(merged), 'city_days': len(city_day), 'means': summary['mean'].to_dict()}
(OUT_DIR / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
daily_wide.plot(marker='o'); plt.ylabel('站点日最高温均值 / °C'); plt.tight_layout()
plt.savefig(OUT_DIR / 'city_comparison.png'); plt.close()
