import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier


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

# PCA
mypca = PCA()
mypca.fit(X_train)

# Varianza por componente
variance = mypca.explained_variance_ratio_
print("\nVarianza que aporta cada componente:")
print(variance)

# Varianza acumulada
print("\nVarianza acumulada:")
acumvar = variance.cumsum()
for i in range(len(acumvar)):
    print(f"{(i+1)} componentes: {acumvar[i]:.8f}")

# PCA con 2 componentes
mypca2 = PCA(n_components=2)
mypca2.fit(X_train)
values_proj2 = mypca2.transform(X_train)

# Reconstrucción y pérdida con 2 componentes
X_projected2 = mypca2.inverse_transform(values_proj2)
loss2 = ((X_train - X_projected2) ** 2).mean()
print("\nProjection loss (2 components):", loss2)

# Gráfica comparativa
plt.figure()

plt.subplot(1,2,1)
plt.title("Datos originales (2 atributos)")
plt.scatter(X_train[:,0], X_train[:,1], marker='o', c=y_train)

plt.subplot(1,2,2)
plt.title("PCA (2 componentes)")
plt.scatter(values_proj2[:,0], values_proj2[:,1], marker='o', c=y_train)

plt.subplots_adjust(right=1.9)
plt.show()

# PCA con 5 componentes
mypca5 = PCA(n_components=5)
mypca5.fit(X_train)
values_proj5 = mypca5.transform(X_train)

# Reconstrucción y pérdida con 5 componentes
X_projected5 = mypca5.inverse_transform(values_proj5)
loss5 = ((X_train - X_projected5) ** 2).mean()

print("\nProjection loss (5 components):", loss5)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

# KNN sin PCA

k_values = [1, 3, 5, 7, 9]

acc_mean_list = []
time_mean_list = []

for k in k_values:
    
    acc_runs = []
    time_runs = []
    
    for i in range(5):
        
        # División entrenamiento / test
        X_train, X_test, y_train, y_test = train_test_split(
            X_readed, y_readed, random_state=i+1
        )
        
        # Balanceo con SMOTE
        sm = SMOTE(random_state=i+1)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        
        # Normalización
        scaler = preprocessing.StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Tiempo de ejecución
        start = time.time()
        
        # Crear modelo
        neigh = KNeighborsClassifier(n_neighbors=k)
        
        # Entrenar modelo
        neigh.fit(X_train, y_train)
        
        # Predicción
        yhat = neigh.predict(X_test)
        
        end = time.time()
        
        # Accuracy
        acc_runs.append(accuracy_score(y_test, yhat))
        
        # Tiempo
        time_runs.append(end - start)
    
    # Medias
    acc_mean = np.mean(acc_runs)
    time_mean = np.mean(time_runs)
    
    acc_mean_list.append(acc_mean)
    time_mean_list.append(time_mean)
    
    print("K =", k)
    print("Accuracy media:", acc_mean)
    print("Tiempo medio:", time_mean)
    print()

# Tabla de resultados
resultados_knn = pd.DataFrame({
    "K": k_values,
    "Accuracy media": acc_mean_list,
    "Tiempo medio": time_mean_list
})

print(resultados_knn)

# Gráfica de exactitud media frente a K
plt.figure()
plt.plot(k_values, acc_mean_list, marker='o')
plt.xlabel("K")
plt.ylabel("Accuracy media")
plt.title("KNN sin PCA: Accuracy media frente a K")
plt.show()

# Gráfica de tiempo medio frente a K
plt.figure()
plt.plot(k_values, time_mean_list, marker='o')
plt.xlabel("K")
plt.ylabel("Tiempo medio de ejecución (s)")
plt.title("KNN sin PCA: Tiempo medio frente a K")
plt.show()

# Selección del mejor K
best_k = resultados_knn.loc[resultados_knn["Accuracy media"].idxmax(), "K"]
print("Mejor K:", best_k)

# KNN con PCA

k_values = [1, 3, 5, 7, 9]

acc_mean_list_pca = []
time_mean_list_pca = []

for k in k_values:
    
    acc_runs = []
    time_runs = []
    
    for i in range(5):
        
        # División entrenamiento / test
        X_train, X_test, y_train, y_test = train_test_split(
            X_readed, y_readed, random_state=i+1
        )
        
        # Balanceo con SMOTE
        sm = SMOTE(random_state=i+1)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        
        # Normalización
        scaler = preprocessing.StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        # PCA
        pca = PCA(n_components=5)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        
        # Tiempo de ejecución
        start = time.time()
        
        # Crear modelo
        neigh = KNeighborsClassifier(n_neighbors=k)
        
        # Entrenar modelo
        neigh.fit(X_train, y_train)
        
        # Predicción
        yhat = neigh.predict(X_test)
        
        end = time.time()
        
        # Accuracy
        acc_runs.append(accuracy_score(y_test, yhat))
        
        # Tiempo
        time_runs.append(end - start)
    
    acc_mean = np.mean(acc_runs)
    time_mean = np.mean(time_runs)
    
    acc_mean_list_pca.append(acc_mean)
    time_mean_list_pca.append(time_mean)
    
    print("K =", k)
    print("Accuracy media PCA:", acc_mean)
    print("Tiempo medio PCA:", time_mean)
    print()


# Tabla de resultados PCA
resultados_knn_pca = pd.DataFrame({
    "K": k_values,
    "Accuracy media PCA": acc_mean_list_pca,
    "Tiempo medio PCA": time_mean_list_pca
})

print(resultados_knn_pca)

# Gráfica de exactitud media frente a K (PCA)
plt.figure()
plt.plot(k_values, acc_mean_list_pca, marker='o')
plt.xlabel("K")
plt.ylabel("Accuracy media")
plt.title("KNN con PCA: Accuracy media frente a K")
plt.show()

# Gráfica de tiempo medio frente a K (PCA)
plt.figure()
plt.plot(k_values, time_mean_list_pca, marker='o')
plt.xlabel("K")
plt.ylabel("Tiempo medio de ejecución (s)")
plt.title("KNN con PCA: Tiempo medio frente a K")
plt.show()

# Selección del mejor K con PCA
best_k_pca = resultados_knn_pca.loc[resultados_knn_pca["Accuracy media PCA"].idxmax(), "K"]
print("Mejor K con PCA:", best_k_pca)

# Arbol de decisión

acc_runs_tree = []
time_runs_tree = []

for i in range(5):
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_readed, y_readed, random_state=i+1
    )
    
    sm = SMOTE(random_state=i+1)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    start = time.time()
    
    tree = DecisionTreeClassifier(random_state=i+1)
    tree.fit(X_train, y_train)
    
    yhat = tree.predict(X_test)
    
    end = time.time()
    
    acc_runs_tree.append(accuracy_score(y_test, yhat))
    time_runs_tree.append(end - start)

print("Árbol de decisión")
print("Accuracy media:", np.mean(acc_runs_tree))
print("Tiempo medio:", np.mean(time_runs_tree))

# Comparación final

print("\nComparación final")

print("\nKNN sin PCA")
print("Mejor K:", best_k)
print("Accuracy:", max(acc_mean_list))
print("Tiempo:", time_mean_list[k_values.index(best_k)])

print("\nKNN con PCA")
print("Mejor K:", best_k_pca)
print("Accuracy:", max(acc_mean_list_pca))
print("Tiempo:", time_mean_list_pca[k_values.index(best_k_pca)])

print("\nÁrbol de decisión")
print("Accuracy:", np.mean(acc_runs_tree))
print("Tiempo:", np.mean(time_runs_tree))
