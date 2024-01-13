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
    DATE_DIED: List
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

@app.post("/model/predict")
async def predict_xgb(data: Dataset_data):
    data = pd.read_json(data.model_dump_json(), orient='list')
    data = data.set_index('index')
    model_prediction = pd.DataFrame(np.round(model.predict(data)))
    model_prediction = model_prediction.reset_index().to_json()
    return model_prediction