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
import time

#### Leer datos de "Binary_no_lineal.csv" ####
Dataset = pd.read_csv("Binary_no_lineal.csv",header=None)
X=Dataset.iloc[:,:2].values
y=Dataset.iloc[:,2].values

#### dibujar distribución de datos ####
X_o = Dataset.loc[:,2]==1
X_x = Dataset.loc[:,2]==0
X_1 = Dataset.loc[X_o].values
X_2 = Dataset.loc[X_x].values
plt.scatter(X_2[:,0],X_2[:,1], color="red",label="0")
plt.scatter(X_1[:,0],X_1[:,1],color="green", label="1")
plt.xlabel("X0")
plt.ylabel("X1")
plt.title("Distribución de datos")
plt.legend()
plt.show()


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.5, random_state=13, stratify=y)

from sklearn.neighbors import KNeighborsClassifier
# construir modelo

#clf = KNeighborsClassifier(n_neighbors=3)
clf = KNeighborsClassifier(n_neighbors=3, weights="distance" , metric="cosine")
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print('accuracy={}'.format(clf.score(X_test,y_test)))

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, 
                               step = 0.05),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, 
                               step = 0.05))
tic = time.time()
plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap(('red', 'green')))
elaps =time.time()-tic

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title("Clasificación no lineal")
plt.xlabel('X0')
plt.ylabel('X1')
plt.legend()
plt.show()

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title("Datos de Prueba")
plt.xlabel('X0')
plt.ylabel('X1')    
plt.legend()
plt.show()    

    