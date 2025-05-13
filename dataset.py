import os
from zipfile import ZipFile
import pandas as pd

# Ruta del zip y carpeta de extracción
zip_path = "/mnt/data/archive.zip"
extract_to = "/mnt/data/dataset"

# Extraer ZIP
with ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

# Mostrar archivos extraídos
archivos = os.listdir(extract_to)
print("Archivos encontrados:", archivos)

# Reemplazá este nombre con el real que veas en el paso anterior
PrimerDataset = os.path.join(extract_to, "Debernardi_et_al_2020_data.csv")
SegundoDataset = os.path.join(extract_to, "Debernardi_et_al_2020_documentation.csv")

df1 = pd.read_excel(PrimerDataset)
df2 = pd.read_excel(SegundoDataset)

print("Primer dataset:")
print(df1.head())

print("Segundo dataset:")
print(df2.head())

df1 = pd.read_excel("ruta/del/archivo1.xlsx")
df2 = pd.read_excel("ruta/del/archivo2.xlsx")

df1.to_csv("archivo1.csv", index=False)
df2.to_csv("archivo2.csv", index=False)
