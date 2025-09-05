from fastapi import FastAPI
from .models import BloodAnalysisInput
import joblib

app = FastAPI(title="PIA - Pancreatic Cancer Prediction API")

# Cargar modelo entrenado
# Ajusta la ruta al archivo .pkl de tu modelo IA
model = joblib.load("ruta/a/tu_modelo.pkl")

@app.post("/predict")
def predict(data: BloodAnalysisInput):
    # Convertir datos a la estructura que espera el modelo
    features = [[
        data.age,
        data.plasma_CA19_9,
        data.creatinine,
        data.LYVE1,
        data.REG1B,
        data.TFF1,
        data.REG1A,
        int(data.sex_F),
        int(data.sex_M),
        data.CEA,
        data.THBS
    ]]

    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}