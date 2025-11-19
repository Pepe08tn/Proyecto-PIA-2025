# README -- Cómo correr el proyecto

## 🚀 Cómo ejecutar el proyecto P.I.A. (Plataforma Inteligente de Asistencia)

Este proyecto utiliza **FastAPI**, **Python**, y modelos de **Machine
Learning / Deep Learning** para la predicción de cáncer de páncreas a
partir de datos estructurados e imágenes médicas.

------------------------------------------------------------------------

## 📦 1. Clonar el repositorio

``` bash
git clone https://github.com/<tu_usuario>/<tu_repositorio>.git
cd <tu_repositorio>
```

------------------------------------------------------------------------

## 🐍 2. Crear el entorno virtual

### En Windows:

``` bash
python -m venv venv
venv\Scripts\activate
```

### En Linux / macOS:

``` bash
python3 -m venv venv
source venv/bin/activate
```

------------------------------------------------------------------------

## 📚 3. Instalar dependencias

``` bash
pip install -r src/requirements.txt
```

------------------------------------------------------------------------

## 🤖 4. Verificar ubicación de modelos

Los modelos deben estar en:

    src/mia_predictor/modelos/

Archivos necesarios: - arbol_decision.pkl\
- random_forest.pkl\
- mia_cnn.pth

------------------------------------------------------------------------

## ▶️ 5. Ejecutar FastAPI

``` bash
uvicorn src.api.api:app --reload
```

Servidor en:

http://127.0.0.1:8001

------------------------------------------------------------------------

## 📘 6. Probar endpoints

Documentación interactiva:

-   Swagger: http://127.0.0.1:8001/docs\
-   ReDoc: http://127.0.0.1:8001/redoc

Endpoints: - `POST /predict` - `POST /predict-image`

------------------------------------------------------------------------

## 🧪 7. Ejemplo respuesta

``` json
{
  "status": "success",
  "prediction": "pancreatic_tumor",
  "probabilities": {
    "normal": 0.03,
    "pancreatic_tumor": 0.97
  }
}
```

------------------------------------------------------------------------

## 🔧 8. Detener servidor

CTRL + C