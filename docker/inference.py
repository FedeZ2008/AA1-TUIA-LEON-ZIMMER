import joblib
import pandas as pd
import numpy as np
import warnings
import os
import sys


# CONFIGURACIÓN
# Archivos
MODEL_PATH = 'pipeline.joblib'
KMEANS_PATH = 'kmeans_region.joblib'
MEDIAN_MAP_PATH = 'mapas_medianas.joblib'
MODE_MAP_PATH = 'mapas_modas.joblib'
SCALER_PATH = 'scaler.joblib'
COORDS_CSV = 'weatherAUS-geo-coordinates.csv'

# Definición de variables
VARIABLES_NUMERICAS = [
    'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
    'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
    'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
    'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm'
]

CAT_VARS = ['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm']


def cargar_artefactos():
    """Carga todos los modelos y diccionarios necesarios."""
    pipeline = joblib.load(MODEL_PATH)
    kmeans = joblib.load(KMEANS_PATH)
    mapas_medianas = joblib.load(MEDIAN_MAP_PATH)
    mapas_modas = joblib.load(MODE_MAP_PATH)
    scaler = joblib.load(SCALER_PATH)
    df_coords = pd.read_csv(COORDS_CSV)
    return pipeline, kmeans, mapas_medianas, mapas_modas, scaler, df_coords


def get_season(dt):
    """asigna estaciones"""
    month = dt.month
    if month in [12, 1, 2]:
        return 'Summer'
    elif month in [3, 4, 5]:
        return 'Autumn'
    elif month in [6, 7, 8]:
        return 'Winter'
    elif month in [9, 10, 11]:
        return 'Spring'


def wind_dir_to_rad(wind_dir):
    """convierte en direcciones de viento a ángulos en radianes"""
    mapping = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
        'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
        'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
        'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }
    return wind_dir.map(mapping).astype(float) * np.pi / 180


def limpieza(df, kmeans, mapas_medianas, mapas_modas, scaler, df_coords):
    
    #generacion de latitudes y longitudes
    df = df.merge(df_coords, left_on='Location', right_on='city', how='left')
    df = df.drop(columns=['city'])


    # asignación de regiones
    X_coords = df[['latitude', 'longitude']]
    df['Region'] = kmeans.predict(X_coords)

    # asignacion de estaciones
    df['Date'] = pd.to_datetime(df['Date'])
    df['Season'] = df['Date'].apply(get_season)

    # Imputacion de valores faltantes
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    for var in VARIABLES_NUMERICAS:
        if var in df.columns and df[var].isnull().any():
            median_map = mapas_medianas.get(var)
            
            if median_map is not None:
                mask = df[var].isnull()
                valores_imputados = df.loc[mask].set_index(['Region', 'Month', 'Day']).index.map(median_map)
                df.loc[mask, var] = valores_imputados
            
    for var in CAT_VARS:
        if var in df.columns and df[var].isnull().any():
            mode_map = mapas_modas.get(var)
            
            if mode_map is not None:
                mask = df[var].isnull()
                valores_imputados = df.loc[mask].set_index(['Region', 'Month', 'Day']).index.map(mode_map)              
                df.loc[mask, var] = valores_imputados
    
    # Limpieza de columnas innecesarias
    df = df.drop(columns=['Day', 'Month','Location','latitude', 'longitude','Date'])            
    

    # Transformación de variables de direccion de viento
    for col in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
        for df_ in [df]:
            rad = wind_dir_to_rad(df_[col])
            df_[f'{col}_sin'] = np.sin(rad)
            df_[f'{col}_cos'] = np.cos(rad)
        df.drop(columns=[col], inplace=True)
        

    # Creacion de variables dummies
    columnas_dummies = ['RainToday','Season','Region']
    df = pd.get_dummies(df, columns=columnas_dummies, prefix=columnas_dummies, drop_first=True, dtype=int)

    df.rename(columns={
        'RainToday_Yes': 'RainToday',
    }, inplace=True)
    # Escalado de variables numéricas
    df[VARIABLES_NUMERICAS] = scaler.transform(df[VARIABLES_NUMERICAS])

    return df

def ejecutar_inferencia(datos):
    try:
        # Carga
        pipeline, kmeans, mapas_medianas, mapas_modas, scaler, df_coords = cargar_artefactos()
        
        # Procesamiento
        df_limpio = limpieza(datos, kmeans, mapas_medianas, mapas_modas, scaler, df_coords)
        
        # Predicción
        prediccion = pipeline.predict(df_limpio)
        return prediccion
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    # Verificamos si el usuario pasó un archivo como argumento
    if len(sys.argv) < 2:
        print("ERROR: Debes indicar el archivo CSV de entrada.")
        print("Uso: python inferencia.py <archivo_datos.csv>")
        sys.exit(1)

    # 1. Obtener la ruta del archivo desde la consola
    input_file = sys.argv[1]
    
    print(f"--- Leyendo archivo: {input_file} ---")

    try:
        # 2. Leer el CSV Input
        df_entrada = pd.read_csv(input_file)
        
        # 3. Ejecutar Inferencia
        predicciones = ejecutar_inferencia(df_entrada)
        
        # 4. Guardar o Mostrar Resultados
        # Opción A: Agregar la predicción al CSV original y guardar uno nuevo
        df_salida = df_entrada.copy()
        df_salida['RainTomorrow_Pred'] = predicciones
        
        output_file = 'predicciones.csv'
        df_salida.to_csv(output_file, index=False)
        
        print(f"Éxito. Predicciones guardadas en '{output_file}'")
        
        # Opcional: Mostrar las primeras filas en consola
        print(df_salida[['Date', 'Location', 'RainTomorrow_Pred']].head())

    except FileNotFoundError:
        print(f"Error: El archivo '{input_file}' no existe.")
    except Exception as e:
        print(f"Error fatal durante la ejecución: {e}")

