# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 13:00:10 2020

Regresión Lineal Simple

Sin usar sklearn  obtener resultados

Estimar línea que representa mejor los datos

@author: Mariko Nakano
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### Leer archivo de dartos 
dataset = pd.read_csv('Data1.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

plt.scatter(X,y,color="blue")
plt.xlabel('X')
plt.ylabel('y')
plt.title("Todos los datos")
plt.show()

#### inicializar parametros ####
from sklearn.linear_model import LinearRegression
modelo = LinearRegression()

modelo.fit(X,y)
y_pred = modelo.predict(X)

th0=modelo.intercept_
th1=modelo.coef_

print("Intercepto: ", th0)
print("Inclinación: ", th1[0])
print("score", modelo.score(X,y))

plt.scatter(X,y,color="blue")
plt.plot(X,th1*X+th0,color="red",label="linea estimada")
plt.xlabel('X')
plt.ylabel('y')
plt.title("Todos los datos")
plt.legend()
plt.show()