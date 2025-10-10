import numpy as np
import pandas as pd
from sklearn.preprocessing import  OneHotEncoder, MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# importar el data set
dataset = pd.read_csv('Fish.csv')

#1. Usando Length1, Length2, Length3, Height y Width Predecir Weight
# Indices X -> [2,...,-1] y -> [1]
#2. Usando Species, Length1, Length2, Length3, Height y Width Predecir Weight

#Analizar Specie de pez influye o no a la predicción de peso de pez.

#'null' --> 0    promedio de especie, 
print(f'{" Visualizacion de dataset ":=^55}')
print(dataset.head())

X = dataset.iloc[:, 2:]
y = dataset.iloc[:, 1].values

# No hay datos categoricos 
mms = MinMaxScaler()
X_completo = mms.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_completo,y,test_size=0.1,random_state=0)

modelo = LinearRegression().fit(X_train, y_train)
score_train = modelo.score(X_train, y_train)
score_test = modelo.score(X_test, y_test)
print(f'Training score primer punto = {score_train*100:7.2f}')
print(f'Test score primer punto= {score_test*100:7.2f}')

X = dataset.iloc[:, [0, 2,3,4,5]]
y = dataset.iloc[:, 1].values

# determine categorical and numerical features
numerical = X.select_dtypes(include=['int64', 'float64'])
categorical = X.select_dtypes(include=['object', 'bool'])

enc = OneHotEncoder()
mms = MinMaxScaler()
onehotlabels = enc.fit_transform(categorical).toarray()
onehotlabels = onehotlabels[:,1:] 

X = numerical.iloc[:,:].values
X_mm = mms.fit_transform(X)
X_completo = np.concatenate((onehotlabels, X_mm), axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X_completo,y,test_size=0.1,random_state=0)

modelo = LinearRegression().fit(X_train, y_train)
score_train = modelo.score(X_train, y_train)
score_test = modelo.score(X_test, y_test)
print(f'Training score segundo punto = {score_train*100:7.2f}')
print(f'Test score segundo punto = {score_test*100:7.2f}')