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
# ## 1. 生成有可解释机制的房屋数据
import numpy as np, pandas as pd, json
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
rng = np.random.default_rng(7); n = 800
houses = pd.DataFrame({'area': rng.uniform(35, 180, n), 'age': rng.integers(0, 45, n), 'district': rng.choice(['north','center','south'], n)})
prices = 80 + houses['area']*2.2 - houses['age']*1.2 + houses['district'].map({'north':20,'center':90,'south':0}) + 0.018*(houses['area']-100)**2 + rng.normal(0,12,n)
houses.loc[rng.choice(n, 35, replace=False), 'area'] = np.nan
houses.assign(price=prices).to_csv(DATA_DIR / 'simulated_houses.csv', index=False)
X_dev, X_test, y_dev, y_test = train_test_split(houses, prices, test_size=0.2, random_state=7)
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.25, random_state=7)


# %%
# ## 2. 让每种列经过适当处理
def preprocessing():
    return ColumnTransformer([
        ('numeric', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), ['area','age']),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['district'])])
def make_models():
    return {'ridge': make_pipeline(preprocessing(), Ridge(alpha=1)),
            'boosting': make_pipeline(preprocessing(), HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15, random_state=7))}
models = make_models()


# %%
# ## 3. 先验证选型，再锁定测试
val_scores = {}
for name, estimator in models.items():
    estimator.fit(X_train, y_train)
    val_scores[name] = float(mean_absolute_error(y_val, estimator.predict(X_val)))
selected = min(val_scores, key=val_scores.get)
predictions = pd.DataFrame({'actual': y_test.to_numpy()}, index=X_test.index)
for name, estimator in models.items():
    estimator.fit(X_dev, y_dev)
    predictions[name] = estimator.predict(X_test)
assert np.isfinite(predictions.to_numpy()).all()


# %%
# ## 4. 看误差，而不只看总分
metrics = {'selected_by_validation': selected, 'validation_mae': val_scores, 'test_mae': {name: float(mean_absolute_error(y_test, predictions[name])) for name in models}}
X_test.join(predictions).to_csv(OUT_DIR / 'predictions.csv', index=False)
(OUT_DIR / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
plt.scatter(y_test, predictions[selected], s=15, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='grey')
plt.xlabel('真实价格 / 教学单位'); plt.ylabel('预测价格'); plt.tight_layout(); plt.savefig(OUT_DIR / 'price_predictions.png'); plt.close(); print(metrics)
