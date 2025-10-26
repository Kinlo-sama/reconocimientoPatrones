# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 13:13:21 2020
RLM_advanced2

Automatizar el proceso de "Eliminación hacia atrás"

@author: Mariko Nakano
"""

import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn.preprocessing import  MinMaxScaler

##### Función que realiza la eliminación hacia atrás automáticamente #####
def backwardElimination(y,X,sl):
    numVars = X.shape[1]   
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, X).fit()     
        maxVar = max(regressor_OLS.pvalues)  
        print(regressor_OLS.summary())   
        if maxVar > sl:  
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j] == maxVar):                    
                    X = X.drop(features[j-1],axis=1)
                    features.remove(features[j-1])


# UCI Machine Leaning Repository [Wine Quality Data Set ]
df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")

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

sl = 0.05
indice = np.linspace(start=0,stop=len(features),num=len(features),dtype=int)
X_Modeled = backwardElimination(y, X, sl)


                
