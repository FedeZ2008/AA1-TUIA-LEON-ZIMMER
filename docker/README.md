# MLOps - Despliegue de Modelo de Predicción de Lluvia

Este directorio contiene los archivos necesarios para construir una imagen de Docker y ejecutar inference.py

## Estructura del Directorio

* **`inference.py`**: Script principal de ejecución. Lee un CSV, procesa los datos y genera predicciones.
* **`Dockerfile`**: Definición de la imagen y el entorno.
* **`requirements_docker.txt`**: Librerías necesarias para la ejecución.
* **`pipeline.joblib`**: Modelo predictivo final.
* **`scaler.joblib`**: Objeto StandardScaler entrenado.
* **`kmeans_region.joblib`**: Modelo K-Means para generar el cluster de regiones.
* **`mapas_medianas.joblib`**: Diccionario de imputación numérica.
* **`mapas_modas.joblib`**: Diccionario de imputación categórica.
* **`weatherAUS-geo-coordinates.csv`**: Base de datos auxiliar de coordenadas.

---

## 🚀 Instrucciones de Uso

### 1. Prerrequisitos
Tener **Docker** instalado y ejecutándose en su máquina.

### 2. Construir la Imagen (Build)
Abra una terminal en esta carpeta (`docker/`) y ejecute:


docker build -t predictor-lluvia .


docker run --rm -v %cd%:/app predictor-lluvia python inferencia.py datos_prueba.csv