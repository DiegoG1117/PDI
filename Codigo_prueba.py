import tensorflow as tf
from tensorflow import keras
import cv2
import os
import numpy as np
import pandas as pd
import glob


data_dir = 'C:/Users/GUSTAVO TOVAR/Documents/GitHub/PDI/Imagenes/'

sub_dirs = os.listdir(data_dir)
sub_dirs.sort()

data = [] 

nuevo_tamano = (603, 804)

for i, class_dir in enumerate(sub_dirs):
    
    class_path = os.path.join(data_dir, class_dir)

    print("Extrayendo Directorios:{}".format(class_dir))

    for archivo_n in glob.glob(os.path.join(class_path, '*.*')):

        print("Procesando archivo: {}".format(archivo_n))

        # Obtener la etiqueta a partir del índice en el bucle
        etiqueta = i

        # Leer la imagen sin redimensionar
        image = cv2.imread(archivo_n, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        
        # Redimensionar la imagen
        image_redimensionada = cv2.resize(image_rgb, nuevo_tamano)

        # Liberar memoria de la imagen original
        del image

        data.append((image_redimensionada, etiqueta))

# Crear un DataFrame con las características y etiquetas
features = pd.DataFrame(data, columns=['Caracteristicas', 'Etiqueta'])

# Guardar el DataFrame como un archivo CSV
features.to_csv('dataset.csv', index=False)

print(features)