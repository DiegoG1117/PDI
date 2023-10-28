import pickle
from matplotlib import pyplot as plt

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