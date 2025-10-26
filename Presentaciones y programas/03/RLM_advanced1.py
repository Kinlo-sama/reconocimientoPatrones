# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 18:11:21 2020

@author: Mariko Nakano
"""

import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import  MinMaxScaler

#UCI Machine Leaning Repository [Wine Quality Data Set ]
df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")

#features = ['fixed acidity','volatile acidity','citric acid','residual sugar',
#        'chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']

features = list(df.columns.values)
features.remove('quality')
x = df[features]
                
y = df[['quality']] 
 
# normalizar los datos
sc_X = MinMaxScaler()
x_norm = sc_X.fit_transform(x)
# Convertir DataFrame
x_norm = pd.DataFrame(x_norm)
x_norm.columns = features

# Agregar 1 en x  --- en caso de sklearn no se requiere
X = sm.add_constant(x_norm)

# Usar statsmodels Ordinary Least Square (OLS)
model = sm.OLS(y, X)
 
# Obtener resultado
results = model.fit()
 
# resultados estadisticos detalles
print(results.summary())

####### generar modelo sin "density" #####
features_sin_density = ['fixed acidity','volatile acidity','citric acid',
                        'residual sugar','chlorides','free sulfur dioxide',
                        'total sulfur dioxide','pH','sulphates','alcohol']

x = df[features_sin_density]
                 
# normalizar los datos
x_norm = sc_X.fit_transform(x)
# Convertir DataFrame
x_norm = pd.DataFrame(x_norm)
x_norm.columns = features_sin_density

# Agregar 1 en x  --- en caso de sklearn no se requiere
X = sm.add_constant(x_norm)

#X.remove('density')
# Usar statsmodels Ordinary Least Square (OLS)
model = sm.OLS(y, X)
 
# Obtener resultado
results = model.fit()
 
# resultados estadisticos detalles
print(results.summary())

####### generar modelo sin "residual_sugar" #####
features_sin_fixed_acidity = ['volatile acidity','citric acid','residual sugar',
        'chlorides','free sulfur dioxide','total sulfur dioxide','pH',
        'sulphates','alcohol']

x = df[features_sin_fixed_acidity]
                 
# normalizar los datos
x_norm = sc_X.fit_transform(x)
# Convertir DataFrame
x_norm = pd.DataFrame(x_norm)
x_norm.columns = features_sin_fixed_acidity

# Agregar 1 en x  --- en caso de sklearn no se requiere
X = sm.add_constant(x_norm)

# Usar statsmodels Ordinary Least Square (OLS)
model = sm.OLS(y, X)
 
# Obtener resultado
results = model.fit()
 
# resultados estadisticos detalles
print(results.summary())

####### generar modelo sin "citric acid" #####
features_sin_sugar = ['volatile acidity','citric acid',
        'chlorides','free sulfur dioxide','total sulfur dioxide','pH','sulphates','alcohol']

x = df[features_sin_sugar]
                 
# normalizar los datos
x_norm = sc_X.fit_transform(x)
# Convertir DataFrame
x_norm = pd.DataFrame(x_norm)
x_norm.columns = features_sin_sugar

# Agregar 1 en x  --- en caso de sklearn no se requiere
X = sm.add_constant(x_norm)

# Usar statsmodels Ordinary Least Square (OLS)
model = sm.OLS(y, X)
 
# Obtener resultado
results = model.fit()
 
# resultados estadisticos detalles
print(results.summary())

####### generar modelo sin "citric acid" #####
features_sin_citric = ['volatile acidity',
        'chlorides','free sulfur dioxide','total sulfur dioxide','pH','sulphates','alcohol']

x = df[features_sin_citric]
                 
# normalizar los datos
x_norm = sc_X.fit_transform(x)
# Convertir DataFrame
x_norm = pd.DataFrame(x_norm)
x_norm.columns = features_sin_citric

# Agregar 1 en x  --- en caso de sklearn no se requiere
X = sm.add_constant(x_norm)

# Usar statsmodels Ordinary Least Square (OLS)
model = sm.OLS(y, X)
 
# Obtener resultado
results = model.fit()
 
# resultados estadisticos detalles
print(results.summary())







