import tensorflow as tf
from tensorflow import keras
import cv2
import os
import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split


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


x = np.array(features.Caracteristicas.tolist())
y = np.array(features.Etiqueta.tolist())

print(x.shape)
print(y.shape)

X = np.asarray(x).astype(dtype=np.float32) 
y = np.asarray(y).astype(dtype=np.int32)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
print(len(X_train))
print(len(y_train))

print(y_train)