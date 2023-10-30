import pickle
import numpy as np
from sklearn.model_selection import train_test_split

# Cargar los datos
with open('C:/Users/GUSTAVO TOVAR/Documents/GitHub/PDI/dataset.pkl', 'rb') as f:
    x, y = pickle.load(f)

print(x.shape)
print(y.shape)

X = np.asarray(x).astype(dtype=np.float32)
y = np.asarray(y).astype(dtype=np.int32)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
print(len(X_train))
print(len(y_train))

print(y_train)

# Guardar los datos de entrenamiento
with open('datos_entrenamiento.pkl', 'wb') as f:
    pickle.dump((X_train, y_train), f)

# Guardar los datos de prueba
with open('datos_prueba.pkl', 'wb') as f:
    pickle.dump((X_test, y_test), f)