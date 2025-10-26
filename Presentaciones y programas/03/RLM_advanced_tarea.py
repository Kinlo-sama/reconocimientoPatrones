# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 16:56:01 2020

@author: user
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import  OneHotEncoder 

# importar el data set
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4].values


# determine categorical and numerical features
numerical = X.select_dtypes(include=['int64', 'float64'])
categorical = X.select_dtypes(include=['object', 'bool'])

enc = OneHotEncoder(categories='auto')
onehotlabels = enc.fit_transform(categorical).toarray()
onehotlabels = onehotlabels[:,1:]  # Evitando co-liealidad

X = numerical.iloc[:,:].values
##### Concatenar onehotlabels con X 
X_completo = np.concatenate((onehotlabels,X),axis=1)

########## Tarea  ##########
''' 
 1. Aplicar statsmodels Ordinary Least Square (OLS) a dataframe
 2. Analizar p-value de cada variables independientes
 3. Aplicar algoritmo de eliminación hacia atrás
 4. Analizar resultados desde el punto de vista de
               R-Cuadrado y R-Cuadrado ajustado
'''              
 