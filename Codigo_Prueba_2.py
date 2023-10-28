import tensorflow as tf
from tensorflow import keras
import cv2
import os
import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
import pickle

data_dir = 'C:/Users/GUSTAVO TOVAR/Documents/GitHub/PDI/Imagenes/'
sub_dirs = os.listdir(data_dir)
sub_dirs.sort()

data = []

IMG_SIZE = (603, 804)  # Tamaño deseado

for i, sub_dir in enumerate(sub_dirs):
    print("Extrayendo Directorios: {}".format(sub_dir))
    
    for archivo_n in glob.glob(os.path.join(data_dir, sub_dir, '*.*')):
        print("Procesando archivo: {}".format(archivo_n))
        etiqueta = i
        # Leer la imagen y cambiar el tamaño
        image = cv2.imread(archivo_n)
        image = cv2.resize(image, IMG_SIZE)
        # Verificar que la imagen tenga la forma adecuada (con tres canales)
        if image.shape[-1] != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        print('Etiqueta de Elemento: {}'.format(etiqueta))
        data.append((image, etiqueta))

# Crear DataFrame
features = pd.DataFrame(data, columns=['Caracteristicas', 'Etiqueta'])

# Guardar el DataFrame en un archivo CSV
features.to_csv('C:/Users/GUSTAVO TOVAR/Documents/GitHub/PDI/conjunto_de_datos.csv', index=False)

x = np.array(features.Caracteristicas.tolist())
y = np.array(features.Etiqueta.tolist())

print(x.shape)
print(y.shape)

with open('C:/Users/GUSTAVO TOVAR/Documents/GitHub/PDI/dataset.pkl', 'wb') as f:
    pickle.dump((x, y), f)