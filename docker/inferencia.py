import joblib
import pandas as pd
import warnings
import sklearn
print(sklearn.__version__)
warnings.simplefilter('ignore')

import logging
from sys import stdout




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