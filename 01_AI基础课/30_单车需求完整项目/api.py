from pathlib import Path
import pandas as pd
import joblib

from fastapi import FastAPI
from pydantic import BaseModel,Field,ConfigDict
class BikeRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    hour:int=Field(ge=0,le=23)
    weekday:int=Field(ge=0,le=6)
    month:int=Field(ge=1,le=12)
    year_index:int=Field(ge=0,le=20)
    holiday:int=Field(ge=0,le=1)
    workingday:int=Field(ge=0,le=1)
    lag24:float=Field(ge=0)
    lag168:float=Field(ge=0)
def create_bike_app(bundle):
    app=FastAPI(title='小时单车需求教学 API')
    @app.get('/health')
    def health():return {'status':'ok','model':bundle['selected']}
    @app.post('/predict')
    def predict(payload:BikeRequest):
        row=pd.DataFrame([payload.model_dump()])[bundle['features']]
        prediction=max(0.,float(bundle['model'].predict(row)[0]))
        return {'predicted_rentals':round(prediction,2),'unit':'次/小时','horizon':'one-hour-ahead rolling'}
    return app

bundle=joblib.load(Path(__file__).resolve().parent/'outputs/model.joblib')
app=create_bike_app(bundle)
