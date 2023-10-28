import pickle
from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow.keras.layers as LK
import tensorflow.keras.models as MK

# Cargar los datos de entrenamiento y prueba
with open('train_data.pkl', 'rb') as f:
    X_train, y_train = pickle.load(f)

with open('test_data.pkl', 'rb') as f:
    X_test, y_test = pickle.load(f)

print("Imagenes de entrenamiento Tamaño: %s,  Etiqueta Tamaño: %s" %(X_train.shape,y_train.shape))
print("Imagenes de prueba Tamaño: %s,  Etiqueta Tamaño: %s" %(X_test.shape,y_test.shape))

for i in range(5):
  plt.subplot(1,5,i+1)
  plt.imshow(X_train[i])
  plt.gca().set_yticklabels([])
  plt.gca().set_xticklabels([])
  plt.gca().set_title("Label" + str(y_train[i]))
plt.show()

#reshape
x_train = X_train.reshape((X_train.shape[0],600,600,3))
x_test = X_test.reshape((X_test.shape[0],600,600,3))
print("Train images size={0}, Label size={1}".format(x_train.shape,x_test.shape))

#normalizacion estandar
x_train, x_test = (x_train-x_train.min())/(x_train.max()-x_train.min()),(x_test-x_test.min())/(x_test.max()-x_test.min())


#Modelo Inception v3

entrada = LK.Input(shape=(803,604,3))

#Modelo de Arquitectura Inception v3
# Bloque Inception 1
conv1x1_1 = LK.Conv2D(32, (1, 1), padding='same', activation='relu')(entrada)

conv3x3_1 = LK.Conv2D(16, (1, 1), padding='same', activation='relu')(entrada)
conv3x3_1 = LK.Conv2D(32, (3, 3), padding='same', activation='relu')(conv3x3_1)

conv5x5_1 = LK.Conv2D(16, (1, 1), padding='same', activation='relu')(entrada)
conv5x5_1 = LK.Conv2D(32, (5, 5), padding='same', activation='relu')(conv5x5_1)

pool_proj_1 = LK.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(entrada)
pool_proj_1 = LK.Conv2D(32, (1, 1), padding='same', activation='relu')(pool_proj_1)

inception_1 = LK.Concatenate(axis=-1)([conv1x1_1, conv3x3_1, conv5x5_1, pool_proj_1])
inception_1 = LK.Dropout(0.2)(inception_1)

# Bloque Inception 2
conv1x1_2 = LK.Conv2D(64, (1, 1), padding='same', activation='relu')(inception_1)

conv3x3_2 = LK.Conv2D(32, (1, 1), padding='same', activation='relu')(inception_1)
conv3x3_2 = LK.Conv2D(64, (3, 3), padding='same', activation='relu')(conv3x3_2)

conv5x5_2 = LK.Conv2D(32, (1, 1), padding='same', activation='relu')(inception_1)
conv5x5_2 = LK.Conv2D(64, (5, 5), padding='same', activation='relu')(conv5x5_2)

pool_proj_2 = LK.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inception_1)
pool_proj_2 = LK.Conv2D(64, (1, 1), padding='same', activation='relu')(pool_proj_2)

inception_2 = LK.Concatenate(axis=-1)([conv1x1_2, conv3x3_2, conv5x5_2, pool_proj_2])
inception_2 = LK.Dropout(0.2)(inception_2)

flat = LK.Flatten()(inception_2)
fc1 = LK.Dense(120, activation='relu')(flat)
#fc1 = LK.Dropout(0.5)(fc1)  
fc2 = LK.Dense(84, activation='relu')(fc1)
salida = LK.Dense(3, activation='softmax')(fc2)

modelo = MK.Model(entrada, salida)

modelo.summary()