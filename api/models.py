from pydantic import BaseModel

# Datos de entrada para el modelo de predicción
class BloodAnalysisInput(BaseModel):
    age: float
    plasma_CA19_9: float
    creatinine: float
    LYVE1: float
    REG1B: float
    TFF1: float
    REG1A: float
    sex_F: bool
    sex_M: bool
    CEA: float
    THBS: float