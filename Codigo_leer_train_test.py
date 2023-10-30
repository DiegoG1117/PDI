import pickle
import numpy as np
from matplotlib import pyplot as plt
import tensorflow.keras.layers as LK
import tensorflow.keras.models as MK
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import Callback
import tensorflow as tf

"""physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No se detectaron GPUs.")"""

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

with open('datos_entrenamiento.pkl', 'rb') as f:
    X_train, y_train = pickle.load(f)

# Cargar los datos de prueba
with open('datos_prueba.pkl', 'rb') as f:
    X_test, y_test = pickle.load(f)

print("Imagenes de entrenamiento Tamaño:%s, Tamaño de las etiquetas%s" %(X_train.shape, y_train.shape))
print("Imagenes de entrenamiento Tamaño:{0}, Tamaño de las etiquetas:{1}".format(X_test.shape, y_test.shape))

from matplotlib import pyplot as plt
plt.imshow(X_train[100], cmap='gray')
plt.gca().set_title('Label: '+str(y_train[103]))
plt.show()

#######################
# Función para callback

######################

entrada = LK.Input(shape=(804, 603, 3))


# Bloque Inception 1
conv1_1 = LK.Conv2D(16, (1,1), activation='relu', padding='same', kernel_regularizer=l2(0.001))(entrada)
conv1_2 = LK.Conv2D(32, (3,3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv1_1)
pool1 = LK.MaxPooling2D(pool_size=(2, 2))(conv1_2)

# Bloque Inception 2
conv2_1 = LK.Conv2D(64, (1,1), activation='relu', padding='same', kernel_regularizer=l2(0.001))(pool1)
conv2_2 = LK.Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv2_1)
pool2 = LK.MaxPooling2D(pool_size=(2, 2))(conv2_2)

# Bloque Inception 3
conv3_1 = LK.Conv2D(128, (1,1), activation='relu', padding='same', kernel_regularizer=l2(0.001))(pool2)
conv3_2 = LK.Conv2D(128, (3,3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv3_1)
pool3 = LK.MaxPooling2D(pool_size=(2, 2))(conv3_2)

# Capas Densas
flat = LK.Flatten()(pool3)
fc1 = LK.Dense(64, activation='relu', kernel_regularizer=l2(0.001))(flat)
salida = LK.Dense(3, activation='softmax')(fc1)

modelo = MK.Model(entrada, salida)

modelo.summary()

modelo.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])

history = modelo.fit(X_train, y_train, epochs=50, batch_size=30, validation_data=(X_test, y_test), verbose=1)
loss, acc = modelo.evaluate(X_test,y_test, verbose=1)
print("Loss:{0} - Accuracy:{1}".format(loss, acc))