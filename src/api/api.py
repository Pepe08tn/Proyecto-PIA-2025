from fastapi import FastAPI
from src.api.models import BloodAnalysisInput
import joblib
import pandas as pd

app = FastAPI(title="PIA - Pancreatic Cancer Prediction API")

# Cargar modelo entrenado y columnas de entrada
model = joblib.load("src/api/model.pkl")
columnas_modelo = joblib.load("src/api/columnas_modelo.pkl")

@app.post("/predict")
def predict(data: BloodAnalysisInput):
    # Convertir datos a DataFrame
    datos = pd.DataFrame([data.dict()])
    datos = datos[columnas_modelo]  # asegurar mismo orden de columnas
    prediction = model.predict(datos)
    return {"prediction": int(prediction[0])}
