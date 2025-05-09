import pandas as pd

def clean_data(df):
    # Eliminar columnas innecesarias
    df = df.dropna(axis=1, how='all')  # Quitar columnas completamente vacías

    # Rellenar o eliminar valores nulos
    df = df.dropna()  # o podés usar df.fillna(df.mean()) si preferís

    # Convertir columnas categóricas si hay
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].astype('category').cat.codes

    return df