# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 00:39:08 2020

Lenar_RM2.py

Regresión Lineal Múltiple
Usando Sklearn 

Revisar coeficientes asignados

@author: Mariko Nakano
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

#### Leer datos ###
dataset = pd.read_csv('BostonHousing.csv')
X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, 13].values


### Estandarizar datos ###
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_std= sc_X.fit_transform(X)
#X_std_test =sc_X.fit(X_test)

### Normalizar datos   [0, 1] ###
from sklearn.preprocessing  import MinMaxScaler
nm_X = MinMaxScaler()
X_nm = nm_X.fit_transform(X)

##### Modelo de regressiion lineal múltiple

modelo = LinearRegression()
modelo.fit(X,Y)

coefficient = modelo.coef_
df_coefficient = pd.DataFrame(coefficient, columns=['Coefs'], 
                              index = ["crime rate", "zone", "Industry",

                              "charles river","nitric oxides", "rooms", "age", 
                              "distance","radial highwatys","tax rate", "ptratio",
                              "blacks","lower status"])
print("score = {}". format(modelo.score(X,Y)))

###### Modelo de regression lineal múltiple con datos estandarizados

modelo.fit(X_std,Y)

coefficient_s = modelo.coef_
df_coefficient_s = pd.DataFrame(coefficient_s, columns=['Coefs'], 
                              index = ["crime rate", "zone", "Industry",

                              "charles river","nitric oxides", "rooms", "age", 
                              "distance","radial highwatys","tax rate", "ptratio",
                              "blacks","lower status"])
print("score_estandarizado = {}". format(modelo.score(X_std,Y)))

modelo.fit(X_nm,Y)

coefficient_n = modelo.coef_
df_coefficient_n = pd.DataFrame(coefficient_n, columns=['Coefs'], 
                              index = ["crime rate", "zone", "Industry",

                              "charles river","nitric oxides", "rooms", "age", 
                              "distance","radial highwatys","tax rate", "ptratio",
                              "blacks","lower status"])
print("score_normalizado = {}". format(modelo.score(X_nm,Y)))

df_coefficient_s.to_csv('coefficients_bostos.csv')