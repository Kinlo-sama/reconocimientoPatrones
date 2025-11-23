# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 20:36:29 2020

SVM-1

Clasificar dos especies de Iris usando 

@author: Mariko Nakano
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from matplotlib.colors import ListedColormap


#### Obtener datos de Iris ####
iris = load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df['target']=iris.target

# Usar solamente dos especies Versicolor y Virginica
# Usar longitud de datos para clasificar
# X = iris.data[50:, 2].reshape(-1,1)   #  matriz de 100 x 1 
# y = iris.target[50:]
# y = y-1   # Versicolor =0  y Virginica =1

###### Usando DataFrame #####

X = df.iloc[50:,2:4].values
y = df.iloc[50:,-1].values -1
Especie=['Versicolor','Virginica']

sc_X = StandardScaler()
X_std = sc_X.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_std,y,random_state=123, stratify=y) # 25% para prueba
### cuando quiere cambiar tasa de entrenamiento y prueba 
###   train_test_split(X_std,y, test_size=0.2, random_state=0)

################ distribución de datos #######

for i, j in enumerate(np.unique(y_train)):
    plt.scatter(X_train[y_train == j, 0], X_train[y_train == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = Especie[j])

plt.title('Iris - entrenamiento')
plt.xlabel('Largo de Patalo')
plt.ylabel('Ancho de Petalo')
plt.legend()
plt.show()

for i, j in enumerate(np.unique(y_test)):
    plt.scatter(X_test[y_test == j, 0], X_test[y_test == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = Especie[j])

plt.title('Iris - test')
plt.xlabel('Largo de Patalo')
plt.ylabel('Ancho de Petalo')
plt.legend()
plt.show()

# Ajustar el SVM en el Conjunto de Entrenamiento

Classifier = SVC(kernel = "linear",random_state = 0)
#Classifier = SVC(kernel = "poly",degree=8, coef0=1.5, random_state = 0)
#Classifier = SVC(kernel = "poly",degree=8, random_state = 0)
Classifier.fit(X_train, y_train)

#### Obtener score de clasificación ###
print("score de entrenamiento = {:.3f}".format(Classifier.score(X_train,y_train)))
print("score de prueba = {:.3f}".format(Classifier.score(X_test,y_test)))

#### Dibujar la distribución (Largo de pétalo y ancho de pétalo) #####

X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, 
                               step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, 
                               step = 0.01))
plt.contourf(X1, X2, Classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.4, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = Especie[j])
plt.title('Iris - train')
plt.xlabel('Largo de Patalo')
plt.ylabel('Ancho de Petalo')
plt.legend()
plt.show()

X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, 
                               step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, 
                               step = 0.01))

plt.contourf(X1, X2, Classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.4, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = Especie[j])

plt.title('Iris test')
plt.xlabel('Largo de Patalo')
plt.ylabel('Ancho de Petalo')
plt.legend()
plt.show()

####### Evaluación global #######

from sklearn.metrics import confusion_matrix
cm_train = confusion_matrix(y_train, Classifier.predict(X_train))
cm_test = confusion_matrix(y_test, Classifier.predict(X_test))

from sklearn.metrics import classification_report
print(classification_report(y_train, Classifier.predict(X_train)))
print(classification_report(y_test, Classifier.predict(X_test)))

