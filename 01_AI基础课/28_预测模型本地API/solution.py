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
# ## 1. 固定特征顺序与目标单位
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

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
rng = np.random.default_rng(28)
X = np.column_stack([rng.uniform(25, 180, 500), rng.integers(1, 6, 500), rng.integers(1, 25, 500)])
y = 600 + 15 * X[:, 0] + 180 * X[:, 1] + 4 * X[:, 2] + rng.normal(0, 70, 500)
model = RandomForestRegressor(n_estimators=60, max_depth=8, random_state=28, n_jobs=1).fit(X[:380], y[:380])
mae = float(mean_absolute_error(y[380:], model.predict(X[380:])))
feature_names = ['area', 'rooms', 'floor']

# %%
# ## 2. 保存与重载一致性
joblib.dump(model, OUT_DIR / 'model.joblib')
loaded = joblib.load(OUT_DIR / 'model.joblib')
assert np.allclose(loaded.predict(X[380:385]), model.predict(X[380:385]))
(OUT_DIR / 'model_metadata.json').write_text(json.dumps({'features': feature_names, 'model_version': 'teaching-v1', 'data': 'synthetic', 'target_unit': '教学货币/月'}, ensure_ascii=False, indent=2), encoding='utf-8')

# %%
# ## 3. 输入校验与响应
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict

class RentRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    area: float = Field(gt=0, le=500)
    rooms: int = Field(ge=1, le=10)
    floor: int = Field(ge=1, le=80)

def create_app(estimator):
    app = FastAPI(title='教学租金预测 API')

    @app.get('/health')
    def health():
        return {'status': 'ok', 'model_version': 'teaching-v1'}

    @app.post('/predict')
    def predict(payload: RentRequest):
        value = max(0.0, float(estimator.predict([[payload.area, payload.rooms, payload.floor]])[0]))
        return {'predicted_rent': round(value, 2), 'unit': '教学货币/月', 'model_version': 'teaching-v1'}
    return app
app = create_app(loaded)

# %%
# ## 4. 通过真实路由测试契约
from fastapi.testclient import TestClient
with TestClient(app) as client:
    health = client.get('/health')
    valid = client.post('/predict', json={'area': 80, 'rooms': 3, 'floor': 5})
    invalid = client.post('/predict', json={'area': -1, 'rooms': 3, 'floor': 5})
    extra = client.post('/predict', json={'area': 80, 'rooms': 3, 'floor': 5, 'secret': 1})
assert health.status_code == valid.status_code == 200 and invalid.status_code == extra.status_code == 422
assert valid.json()['predicted_rent'] >= 0
(OUT_DIR / 'api_responses.json').write_text(json.dumps({'valid': valid.json(), 'invalid_status': invalid.status_code, 'extra_field_status': extra.status_code}, ensure_ascii=False, indent=2), encoding='utf-8')
barplot(['正常预测', '负面积', '额外字段'], [valid.status_code, invalid.status_code, extra.status_code], '实际 HTTP 状态码')
finish({'test_mae': mae, 'valid_request_status': valid.status_code, 'invalid_request_status': invalid.status_code, 'serialization_match': True, 'sample_prediction': valid.json()})