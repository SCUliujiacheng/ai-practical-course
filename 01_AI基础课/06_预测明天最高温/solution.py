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
# ## 1. 对齐到每日时间轴
import pandas as pd, numpy as np, json
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
daily = pd.read_csv(COURSE_ROOT / 'assets/data/jena_daily.csv', parse_dates=['date']).sort_values('date')
assert daily['date'].is_unique
daily = daily.set_index('date').asfreq('D')
assert len(daily) > 365
print(daily.head(), daily.isna().sum())


# %%
# ## 2. 把预测问题变成表格
def make_weather_table(frame, window=7, target='temp_max'):
    table = frame[['temp_max', 'temp_min', 'temp_mean', 'pressure', 'humidity']].copy()
    for lag in range(1, window + 1):
        table[f'temp_lag_{lag}'] = frame['temp_max'].shift(lag)
    table['target'] = frame[target].shift(-1)
    table['forecast_date'] = frame.index + pd.Timedelta(days=1)
    return table.dropna()
table = make_weather_table(daily)
features = [c for c in table.columns if c not in ['target', 'forecast_date']]
a, b = int(len(table)*0.6), int(len(table)*0.8)
train, validation, test = table.iloc[:a], table.iloc[a:b], table.iloc[b:]
assert train['forecast_date'].max() < validation['forecast_date'].min()
assert validation['forecast_date'].max() < test['forecast_date'].min()


# %%
# ## 3. 只用验证集选择正则化强度
validation_scores = {}
for alpha in [0.1, 1.0, 10.0, 100.0]:
    candidate = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    candidate.fit(train[features], train['target'])
    validation_scores[alpha] = float(mean_absolute_error(validation['target'], candidate.predict(validation[features])))
best_alpha = min(validation_scores, key=validation_scores.get)
model = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))
development = table.iloc[:b]
model.fit(development[features], development['target'])
prediction = model.predict(test[features])
baseline = test['temp_max'].to_numpy()


# %%
# ## 4. 评价与失败日期
results = pd.DataFrame({'date': test['forecast_date'].to_numpy(), 'actual': test['target'].to_numpy(), 'persistence': baseline, 'ridge': prediction})
results['absolute_error'] = abs(results['actual'] - results['ridge'])
metrics = {'rows': len(table), 'train': len(train), 'validation': len(validation), 'test': len(test), 'best_alpha': best_alpha, 'validation_mae': validation_scores, 'persistence_mae': float(mean_absolute_error(test['target'], baseline)), 'ridge_mae': float(mean_absolute_error(test['target'], prediction))}
results.to_csv(OUT_DIR / 'predictions.csv', index=False)
results.nlargest(10, 'absolute_error').to_csv(OUT_DIR / 'largest_errors.csv', index=False)
(OUT_DIR / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
results.set_index('date')[['actual', 'persistence', 'ridge']].tail(90).asfreq('D').plot()
plt.ylabel('最高温度 / °C'); plt.tight_layout(); plt.savefig(OUT_DIR / 'forecast.png'); plt.close(); print(metrics)
