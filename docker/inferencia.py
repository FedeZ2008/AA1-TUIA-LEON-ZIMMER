import joblib
import pandas as pd
import warnings
import sklearn
print(sklearn.__version__)
warnings.simplefilter('ignore')

import logging
from sys import stdout

# --- CONFIGURACIÓN ---
# Nombres de los archivos que deben estar en la misma carpeta
MODEL_PATH = 'pipeline.joblib'
KMEANS_PATH = 'kmeans_region.joblib'
MEDIAN_MAP_PATH = 'climatologia_medianas.joblib'
MODE_MAP_PATH = 'mapas_modas.joblib'

# Definición de variables (Las mismas que usaste en train)
VARIABLES_NUMERICAS = [
    'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
    'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
    'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
    'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm'
]

CAT_VARS = ['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm']


def cargar_artefactos():
    """Carga todos los modelos y diccionarios necesarios."""
    required_files = [MODEL_PATH, KMEANS_PATH, MEDIAN_MAP_PATH, MODE_MAP_PATH]
    # Cargar archivos
    pipeline = joblib.load(MODEL_PATH)
    kmeans = joblib.load(KMEANS_PATH)
    mapas_medianas = joblib.load(MEDIAN_MAP_PATH)
    mapas_modas = joblib.load(MODE_MAP_PATH)
    
    return pipeline, kmeans, mapas_medianas, mapas_modas



def limpieza(df):
    
    #generacion de latitudes y longitudes, drop de city
    df_coords = pd.read_csv('weatherAUS-geo-coordinates.csv')
    df = df.merge(df_coords, left_on='Location', right_on='city', how='left')
    df = df.drop(columns=['city'])
    kmeans = joblib.load('kmeans_region.plk')
    X = df[['latitude', 'longitude']]
    df['region'] = kmeans.predict(X)
    df['Date'] = pd.to_datetime(df['Date'])

    # asignacion de estaciones
    def get_season(dt):
        month = dt.month
        if month in [12, 1, 2]:
            return 'Summer'
        elif month in [3, 4, 5]:
            return 'Autumn'
        elif month in [6, 7, 8]:
            return 'Winter'
        elif month in [9, 10, 11]:
            return 'Spring'
    df['Season'] = df['Date'].apply(get_season)

    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    # 3. IMPUTACIÓN NUMÉRICA (Por Region, Mes, Dia)
    for var in VARIABLES_NUMERICAS:
        if var in df.columns and df[var].isnull().any():
            # Recuperamos el mapa para esta variable
            median_map = mapas_medianas.get(var)
            
            if median_map is not None:
                mask = df[var].isnull()
                # Buscamos en el mapa usando el índice múltiple
                valores_imputados = df.loc[mask].set_index(['Region', 'Month', 'Day']).index.map(median_map)
                
                # Rellenamos
                df.loc[mask, var] = valores_imputados
              

    # 4. IMPUTACIÓN CATEGÓRICA (Por Region, Mes, Dia)
    for var in CAT_VARS:
        if var in df.columns and df[var].isnull().any():
            mode_map = mapas_modas.get(var)
            
            if mode_map is not None:
                mask = df[var].isnull()
                valores_imputados = df.loc[mask].set_index(['Region', 'Month', 'Day']).index.map(mode_map)
                
                df.loc[mask, var] = valores_imputados
    df = df.drop(columns=['Day', 'Month','Location','latitude', 'longitude','Date'])            
    
    # Función para convertir direcciones de viento a ángulos en radianes
    def wind_dir_to_rad(wind_dir):
    
        mapping = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }
        return wind_dir.map(mapping).astype(float) * np.pi / 180

    for col in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
        for df_ in [df]:
            rad = wind_dir_to_rad(df_[col])
            df_[f'{col}_sin'] = np.sin(rad)
            df_[f'{col}_cos'] = np.cos(rad)
        df.drop(columns=[col], inplace=True)
        

    # Creamos variables dummies
    columnas_dummies = ['RainToday','Season','Region']
    df = pd.get_dummies(df, columns=columnas_dummies, prefix=columnas_dummies, drop_first=True, dtype=int)

    df.rename(columns={
        'RainToday_Yes': 'RainToday',
    }, inplace=True)

    return df







logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logFormatter = logging.Formatter("%(asctime)s %(levelname)s %(filename)s: %(message)s")
consoleHandler = logging.StreamHandler(stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

pipeline = joblib.load('pipeline.pkl')

logger.info('loaded pipeline')

df_input = pd.read_csv('/files/input.csv')

logger.info('loaded input')

print(df_input.head())

output = pipeline.predict(df_input)

logger.info('made predictions')

pd.DataFrame(output, columns=['MEDV_predicted']).to_csv('/files/output.csv', index=False)

logger.info('saved output')