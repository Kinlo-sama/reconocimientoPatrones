# -*- coding: utf-8 -*-
"""
Spyder Editor

Mariko Nakano

Clasificador KNN

This is a temporary script file.
"""

### Leer datos y Visualizar datos ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Binary_lineal.csv",header=None)
dataset.columns = ['exam1','exam2','pass']
X=dataset.iloc[:,:2].values
y=dataset.iloc[:,2].values

#### dibujar distribución de datos ####

X_pass=dataset[dataset['pass']==1]
X_no_pass = dataset[dataset['pass']==0]
X_pass_data = X_pass.iloc[:,:2].values
X_no_pass_data = X_no_pass.iloc[:,:2].values
plt.scatter(X_pass_data[:,0],X_pass_data[:,1],color="green",label="admitido")
plt.scatter(X_no_pass_data[:,0],X_no_pass_data[:,1], color="red",label="No admitido")
plt.xlabel("Examen 1")
plt.ylabel("Examen 2")
plt.title("Distribución de admitidos y no admitidos")
plt.legend()
plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# construir modelo

# parametros

metrics =["euclidean", "cosine", "manhattan"]
weights = ["uniform", "distance"]
n_neighbors = [1,3,5,7,9]

parameters = {'metric': metrics,
              'weights': weights,
              'n_neighbors': n_neighbors
              }

model_tuning = GridSearchCV(
    estimator =KNeighborsClassifier(),
    param_grid = parameters)


model_tuning.fit(X_train, y_train)

best_param = model_tuning.best_params_
best_knn = model_tuning.best_estimator_


y_pred = best_knn.predict(X_test)
# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, 
                               step = 0.1),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, 
                               step = 0.1))
plt.contourf(X1, X2, best_knn.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Admitido 1 - No admitido 0')
plt.xlabel('Examen 1')
plt.ylabel('Examen 2')
plt.legend()
plt.show()

#######
knn_best_params = KNeighborsClassifier(n_neighbors=7, metric ="manhattan", weights="distance")
modelo = knn_best_params.fit(X_train, y_train)

y_pred2 = modelo.predict(X_test)
cm2 = confusion_matrix(y_test, y_pred2)
