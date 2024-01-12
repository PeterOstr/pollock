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