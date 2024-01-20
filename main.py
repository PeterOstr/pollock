from fastapi import FastAPI
import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import List
from fastapi.staticfiles import StaticFiles
from sklearn.metrics import f1_score, roc_auc_score


import joblib

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

class Dataset_data(BaseModel):
    index: List
    USMER: List
    MEDICAL_UNIT: List
    SEX: List
    PATIENT_TYPE: List
    # DATE_DIED: List
    INTUBED: List
    PNEUMONIA: List
    AGE: List
    PREGNANT: List
    DIABETES: List
    COPD: List
    ASTHMA: List
    INMSUPR: List
    HIPERTENSION: List
    OTHER_DISEASE: List
    CARDIOVASCULAR: List
    OBESITY: List
    RENAL_CHRONIC: List
    TOBACCO: List
    CLASIFFICATION_FINAL: List
    ICU: List


model_logreg = joblib.load('model_logreg.joblib')
model_knn = joblib.load('model_knn.joblib')
model_cat = joblib.load('model_catb.joblib')
model_xgb = joblib.load('model_xgb.joblib')
model_nb = joblib.load('model_nb.joblib')
model_lgbm = joblib.load('model_lgbm.joblib')

@app.post("/model/predict_xgb")
async def predict_xgb(data: Dataset_data):
    data = pd.read_json(data.model_dump_json(), orient='list')
    data = data.set_index('index')
    model_prediction = pd.DataFrame(np.round(model_xgb.predict(data)))
    model_prediction = model_prediction.reset_index().to_json()
    return model_prediction

@app.post("/model/predict_catb")
async def predict_catb(data: Dataset_data):
    data = pd.read_json(data.model_dump_json(), orient='list')
    data = data.set_index('index')
    model_prediction = pd.DataFrame(np.round(model_cat.predict(data)))
    model_prediction = model_prediction.reset_index().to_json()
    return model_prediction

@app.post("/model/predict_lgbm")
async def predict_lgbm(data: Dataset_data):
    data = pd.read_json(data.model_dump_json(), orient='list')
    data = data.set_index('index')
    model_prediction = pd.DataFrame(np.round(model_lgbm.predict(data)))
    model_prediction = model_prediction.reset_index().to_json()
    return model_prediction

@app.post("/model/predict_logreg")
async def predict_logreg(data: Dataset_data):
    data = pd.read_json(data.model_dump_json(), orient='list')
    data = data.set_index('index')
    model_prediction = pd.DataFrame(np.round(model_logreg.predict(data)))
    model_prediction = model_prediction.reset_index().to_json()
    return model_prediction

@app.post("/model/predict_knn")
async def predict_knn(data: Dataset_data):
    data = pd.read_json(data.model_dump_json(), orient='list')
    data = data.set_index('index')
    model_prediction = pd.DataFrame(np.round(model_knn.predict(data)))
    model_prediction = model_prediction.reset_index().to_json()
    return model_prediction


@app.post("/model/predict_nb")
async def predict_nb(data: Dataset_data):
    data = pd.read_json(data.model_dump_json(), orient='list')
    data = data.set_index('index')
    model_prediction = pd.DataFrame(np.round(model_nb.predict(data)))
    model_prediction = model_prediction.reset_index().to_json()
    return model_prediction

@app.post("/model/predict_ens")
async def predict_ens(data: Dataset_data):
    data = pd.read_json(data.model_dump_json(), orient='list')
    data = data.set_index('index')

    model_prediction_xgb = pd.DataFrame(np.round(model_xgb.predict(data)))
    model_prediction_catb = pd.DataFrame(np.round(model_cat.predict(data)))
    model_prediction_lgbm = pd.DataFrame(np.round(model_lgbm.predict(data)))
    model_prediction_logreg = pd.DataFrame(np.round(model_logreg.predict(data)))
    model_prediction_nb = pd.DataFrame(np.round(model_nb.predict(data)))
    model_prediction_knn = pd.DataFrame(np.round(model_knn.predict(data)))

    # averaging predictions
    ensemble_prediction = (
                                  model_prediction_xgb +
                                  model_prediction_catb +
                                  model_prediction_lgbm +
                                  model_prediction_logreg +
                                  model_prediction_nb +
                                  model_prediction_knn
                          ) / 6  # for quantity of models

    # to JSON
    ensemble_prediction_json = ensemble_prediction.reset_index().to_json()

    return ensemble_prediction_json
