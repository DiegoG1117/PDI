import tensorflow as tf
from tensorflow import keras
import cv2
import os
import numpy as np
import pandas as pd
import glob


def redimensionar_imagen(imagen_path, nuevo_tamano):
    try:
        img = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
        img_redimensionada = cv2.resize(img, nuevo_tamano)
        img_redimensionada = np.expand_dims(img_redimensionada, axis=-1)
        return img_redimensionada
    except Exception as e:
        print(f"Error al redimensionar la imagen: {e}")
        return None

# Definir el nuevo tamaño de las imágenes
nuevo_tamano = (3024, 4032)  # Ancho x Alto

# Directorio donde están tus imágenes
data_dir = 'C:/Users/GUSTAVO TOVAR/Documents/GitHub/PDI/'

# Obtener una lista de carpetas en el directorio
data_dir2 = os.listdir(data_dir)

# Ordenar la lista de forma ascendente
data_dir2.sort()

data = []

for i, sub_dir in enumerate(data_dir2):
    # Comprobar si el subdirectorio es una carpeta
    if os.path.isdir(os.path.join(data_dir, sub_dir)):
        print("Extrayendo Directorio: {}".format(sub_dir))

        # Obtener la ruta completa de la carpeta de imágenes
        sub_dir_path = os.path.join(data_dir, sub_dir)

        for archivo_n in glob.glob(os.path.join(sub_dir_path, '*.*')):
            print("Procesando archivo: {}".format(archivo_n))

            # Obtener la etiqueta a partir del nombre de la carpeta
            etiqueta = i % 3  # Asegura que las etiquetas estén en el rango 0 a 2

            # Redimensionar la imagen
            image = redimensionar_imagen(archivo_n, nuevo_tamano)

            if image is not None:
                data.append((image, etiqueta))

# Crear un DataFrame con las características y etiquetas
features = pd.DataFrame(data, columns=['Caracteristicas', 'Etiqueta'])

# Guardar el DataFrame como un archivo CSV
features.to_csv('dataset.csv', index=False)

print(features)
