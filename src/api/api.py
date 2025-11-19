import sys
import os

# Asegura que src/ quede en PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException
from src.api.models import BloodAnalysisInput
import joblib
import pandas as pd
import shutil

# ---------------------------------
# APP
# ---------------------------------
app = FastAPI(title="PIA - Pancreatic Cancer Prediction API")

# ---------------------------------
# PREDICCIÓN ANÁLISIS DE SANGRE
# ---------------------------------

model = joblib.load("src/api/model.pkl")
columnas_modelo = joblib.load("src/api/columnas_modelo.pkl")

@app.post("/predict")
def predict(data: BloodAnalysisInput):
    datos = pd.DataFrame([data.dict()])
    datos = datos[columnas_modelo]
    prediction = model.predict(datos)
    return {"prediction": int(prediction[0])}

# ---------------------------------
# PREDICCIÓN IMÁGENES
# ---------------------------------

from src.mia_predictor.prediccion import predict as predict_image_model

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):

    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Formato de imagen no soportado")

    temp_path = f"temp_{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = predict_image_model(temp_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la IA de imágenes: {str(e)}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {
        "status": "success",
        "prediction": result["predicted_class"],
        "probabilities": result["probabilities"]
    }