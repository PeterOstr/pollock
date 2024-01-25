FROM python:3.10-slim

WORKDIR /app



COPY covid_front_app.py .
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "covid_front_app.py", "--server.port=8501", "--server.address=0.0.0.0"]