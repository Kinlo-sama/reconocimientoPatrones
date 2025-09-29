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

#### DEfinir funciones
def cost_func(X,y,th0, th1):
    error=0
    for i in range(len(X)):
        y_pred=th1*X[i] + th0
        error+=(y[i] - y_pred)*(y[i] - y_pred)
    return error/(2*len(X))        

def up_date(X,y,th0,th1,alfa):
    grad_0 = 0
    grad_1 = 0
    for i in range(len(X)):
        y_pred=th1*X[i] + th0
        grad_0+=(y_pred-y[i])
        grad_1+=(y_pred - y[i])*X[i]
    grad_0/=len(X)
    grad_1/=len(X)
    th0 = th0 - alfa*grad_0
    th1 = th1 - alfa*grad_1
    return th0, th1
        
    
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
th0_mio = -4
th1_mio = 1.2
alfa = 0.01
num_iter = 1000
cost=[]
for  i in range(num_iter):
    th0_mio,th1_mio = up_date(X,y,th0_mio,th1_mio,alfa)
    cost.append(cost_func(X,y,th0_mio,th1_mio))
    
print("Intercepto: ", th0_mio)
print("Inclinación: ", th1_mio)
print('El valor de coste final=', cost[num_iter-1])
# Coeficientes de determinación R^2
score_mio = 1-2*cost[num_iter-1]/y.var() # 1-2*m*cost[num_iter-1]/(m*y.var())
print("Score:", score_mio)

plt.plot(cost)
plt.title("Aprendizaje -- Función de Costo")
plt.xlabel("Iteración")
plt.ylabel("Función de Costo")
plt.show()   

plt.scatter(X,y,color="blue")
plt.plot(X,th1_mio*X+th0_mio,color="red",label="linea estimada")
plt.xlabel('X')
plt.ylabel('y')
plt.title("Todos los datos")
plt.legend()
plt.show()