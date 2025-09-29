#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 12:07:43 2019

Este programa es obtener del archivo un datos de csv.

Generar Regresión Lineal simple para predecir salario a partir de años de experiencia
"""

# Regresión Lineal Simple

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Importar el data set
dataset = pd.read_csv('Salary_Data.csv')
# Importar el data set sin encabezado
#dataset = pd.read_csv('Salary_Data_sinH.csv')  # error
#dataset = pd.read_csv('Salary_Data_sinH.csv',header=None) # pone '0','1',.. como nombre de columna
# Importar el data set separado con espacio
#dataset = pd.read_csv('Salary_Data_espacio.csv',header=None, sep=' ')

# Obtener variable independiente y variable dependiente desde Dataframe 
# Usar metodo loc
#X = dataset.loc[:,'YearsExperience'].values
#y = dataset.loc[:,'Salary'].values
# Usar metodo iloc --- es más conveniente 
X = dataset.iloc[:, :-1].values  # puede ser dataset.iloc[:,0], :-1 -> todos excepto último 
y = dataset.iloc[:, 1].values    # puede ser dataset.iloc[:,-1]

X = dataset.iloc[:, :-1]  # X es dataframe --- 

# Dividir el dataset en conjunto de entrenamiento y conjunto de testing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, 
                                                    random_state = 0)

# Escalado de variables
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

#Crear modelo de Regresión Lienal Simple con el conjunto de entrenamiento

regression = LinearRegression()
regression.fit(X_train, y_train)

# Predecir el conjunto de test
y_pred = regression.predict(X_test)
score_train = regression.score(X_train,y_train)
score_test = regression.score(X_test,y_test)

print("Score de prediccion para training set = {:.3f}".format(score_train))
print("Score de predicción para test set = {:.3f}".format(score_test))

#### Obtener parametrod s línea
print("Intercept = ",regression.intercept_)
print("Incrinación = ",regression.coef_)
#
# Visualizar los resultados de entrenamiento
plt.scatter(X_train, y_train, color = "red", label="training set")
plt.scatter(X_test, y_test, color = "green", label="test set")
plt.scatter(X_test, y_pred, color="cyan", label="predicted")
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Sueldo vs Años de Experiencia (Conjunto de datos)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo (en $)")
plt.legend()
plt.show()

# Visualizar los resultados de test


