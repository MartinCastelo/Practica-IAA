import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn import preprocessing
from sklearn.decomposition import PCA

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss


df = pd.read_csv("ai4i2020.csv") # cargar dataset

print(df.head()) # mostrar las primeras 5 filas

print(df.shape) # mostrar tamaño (10000, 14)

df = df.drop(["UDI",  "Product ID",  "TWF",  "HDF",  "PWF",  "OSF",  "RNF"], axis=1) # elimino columnas identificativas y los tipos de fallos

print(df.isnull().sum()) # suma los valores perdidos de cada columna, no hay en este caso

df["Type"] = df["Type"].map({"L": 0, "M": 1, "H": 2})  # convertir Type a valores numéricos

feature_df = df[['Type', 'Air temperature [K]', 'Process temperature [K]','Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']] # datos de entrada

X_readed = np.asarray(feature_df) # convertir de pandas a array de NumPy
print(X_readed[0:5]) # mostramos las primeras 5 filas
print(feature_df.shape) # mostrar dimension

y_readed = np.asarray(df['Machine failure']) # datos de salida y convertir a array
print(y_readed[0:5])

X_train, X_test, y_train, y_test = train_test_split(X_readed, y_readed, random_state = 1)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# comprobar desbalanceo antes del oversampling
unique, counts = np.unique(y_train, return_counts=True)
print("Clases antes del balanceo:", dict(zip(unique, counts)))

# aplicar oversampling con SMOTE
sm = SMOTE(random_state=1)
X_train, y_train = sm.fit_resample(X_train, y_train)

# comprobar balanceo después de SMOTE
print('Train set balanceado:', X_train.shape, y_train.shape)

unique, counts = np.unique(y_train, return_counts=True)
print("Clases después del balanceo:", dict(zip(unique, counts)))

# Normalización de los datos
scaler = preprocessing.StandardScaler()

# Ajuste del modelo con los datos de entrenamiento
scaler.fit(X_train)

# Transformación de los datos de entrenamiento
X_train = scaler.transform(X_train)

# Transformación de los datos de test
X_test = scaler.transform(X_test)

# Visualización de los primeros datos transformados
print(X_train[0:5])
