FROM python 3.10-slim
WORKDIR /app

COPY main.py .
COPY requirements.txt .

COPY model_catb.joblib .
COPY model_knn.joblib .
COPY model_lgbm.joblib .
COPY model_logreg.joblib .
COPY model_nb.joblib .
COPY model_xgb.joblib .

RUN pip install -r requirements.txt

ENTRYPOINT ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]