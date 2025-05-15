# notebooks/01_analisis_inicial.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "C:\Users\pc\OneDrive\Escritorio\Debernardi et al 2020 data.csv")
df = pd.read_csv(DATA_PATH)


path = "johnjdavisiv/urinary-biomarkers-for-pancreatic-cancer"
Pancreas : pd.DataFrame = pd.read_csv(path)
Pancreas


print("🔍 Primeras filas del dataset:")
print(df.head(), "\n")

# 2. Dimensiones del dataset
print(f"📏 Dimensiones: {df.shape[0]} filas y {df.shape[1]} columnas\n")

# 3. Tipos de datos
print("📊 Tipos de datos por columna:")
print(df.dtypes, "\n")

# 4. Valores faltantes
print("❗ Valores nulos por columna:")
print(df.isnull().sum(), "\n")

# 5. Estadísticas descriptivas
print("📈 Estadísticas descriptivas:")
print(df.describe(include='all'), "\n")

# 6. Distribución de la variable objetivo (si existe)
if 'Diagnosis' in df.columns or 'diagnosis' in df.columns:
    target_col = 'Diagnosis' if 'Diagnosis' in df.columns else 'diagnosis'
    print(f"🎯 Distribución de clases en '{target_col}':")
    print(df[target_col].value_counts(), "\n")

    sns.countplot(data=df, x=target_col, palette="pastel")
    plt.title(f"Distribución de la variable objetivo: {target_col}")
    plt.tight_layout()
    plt.show()

# 7. Correlación entre variables numéricas
numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns
if len(numerical_cols) > 1:
    corr = df[numerical_cols].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Mapa de calor de correlaciones")
    plt.tight_layout()
    plt.show()
