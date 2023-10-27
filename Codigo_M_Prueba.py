import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split

features = pd.read_csv('dataset.csv')

x = features['Caracteristicas'].values
y = features['Etiqueta'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("Tamaño del conjunto de entrenamiento:", len(x_train))
print("Tamaño del conjunto de prueba:", len(x_test))

print(x.shape)
print(y.shape)