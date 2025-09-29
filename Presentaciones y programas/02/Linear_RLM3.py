# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 15:30:11 2020


Regresión lineal Multiple
Manejo de datos categóricos 
prueba de OneHotEncoder

@author: Mariko Nakano
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import  OneHotEncoder , MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/abalone.csv'
dataframe = pd.read_csv(url, header=None)
dataframe.columns = ['Type','Length','Diameter','Height','Whole W',
                     'Shucked W','Viscera W','Shell W', 'Ring']
# split into inputs and outputs

X, y = dataframe.drop('Ring', axis=1), dataframe['Ring']

# determine categorical and numerical features
numerical = X.select_dtypes(include=['int64', 'float64'])
categorical = X.select_dtypes(include=['object', 'bool'])

#enc = OneHotEncoder(categories='auto')
# enc = OneHotEncoder()
# onehotlabels = enc.fit_transform(categorical).toarray()
# onehotlabels = onehotlabels[:,1:]  # Evitando co-liealidad

lb =LabelBinarizer()
lb_data =lb.fit_transform(categorical)
onehotlabels=lb_data[:,:2]  # lb_data[:,1:]

X = numerical.iloc[:,:].values
##### Concatenar onehotlabels con X 
X_completo = np.concatenate((onehotlabels,X),axis=1)
y = y.iloc[:].values

sc_X = MinMaxScaler()
X = sc_X.fit_transform(X_completo)

X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.1,random_state=0)

''' Aplicar el modelo de regresión lineal multiple '''
modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
y_pred = np.round(y_pred)
score_train = modelo.score(X_train,y_train)
score_test = modelo.score(X_test,y_test)
print('training score = {:7.3f}'.format(score_train))
print('test score = {:7.3f}'.format(score_test))
 