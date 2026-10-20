from pathlib import Path
import joblib

from fastapi import FastAPI
from pydantic import BaseModel,Field,ConfigDict
class RentRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    area: float=Field(gt=0,le=500)
    rooms: int=Field(ge=1,le=10)
    floor: int=Field(ge=1,le=80)
def create_app(estimator):
    app=FastAPI(title='教学租金预测 API')
    @app.get('/health')
    def health():return {'status':'ok','model_version':'teaching-v1'}
    @app.post('/predict')
    def predict(payload:RentRequest):
        value=max(0.,float(estimator.predict([[payload.area,payload.rooms,payload.floor]])[0]))
        return {'predicted_rent':round(value,2),'unit':'教学货币/月','model_version':'teaching-v1'}
    return app

model=joblib.load(Path(__file__).resolve().parent/'outputs/model.joblib')
app=create_app(model)
