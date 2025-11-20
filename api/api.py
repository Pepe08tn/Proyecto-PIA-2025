import sys
import os

# ---------------------------------
# CONFIGURACIÓN DE RUTAS
# ---------------------------------

# Directorio del archivo api.py → api/
API_DIR = os.path.dirname(os.path.abspath(__file__))

# Directorio raíz del proyecto → PROYECTO-PIA-2025/
ROOT_DIR = os.path.dirname(API_DIR)

# ⭐⭐ AGREGA EL RAÍZ AL PYTHONPATH ⭐⭐
# Con esto Python puede importar correctamente mia_predictor/
sys.path.append(ROOT_DIR)

# ---------------------------------
# IMPORTS
# ---------------------------------

from fastapi import FastAPI, UploadFile, File, HTTPException
from api.models import BloodAnalysisInput
from mia_predictor.prediccion import predict as predict_image_model

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

model_path = os.path.join(API_DIR, "model.pkl")
columns_path = os.path.join(API_DIR, "columnas_modelo.pkl")

model = joblib.load(model_path)
columnas_modelo = joblib.load(columns_path)

@app.post("/predict")
def predict(data: BloodAnalysisInput):
    datos = pd.DataFrame([data.dict()])
    datos = datos[columnas_modelo]
    prediction = model.predict(datos)
    return {"prediction": int(prediction[0])}

# ---------------------------------
# PREDICCIÓN DE IMÁGENES
# ---------------------------------

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):

    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Formato de imagen no soportado")

    temp_path = os.path.join(ROOT_DIR, f"temp_{file.filename}")

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
