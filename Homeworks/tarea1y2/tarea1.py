import numpy as np
import pandas as pd
from sklearn.preprocessing import  OneHotEncoder , MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4].values

numerical = X.select_dtypes(include=['int64', 'float64'])
categorical = X.select_dtypes(include=['object', 'bool'])

enc = OneHotEncoder()
mms = MinMaxScaler()
onehotlabels = enc.fit_transform(categorical).toarray()

onehotlabels = onehotlabels[:,1:]

X = numerical.iloc[:,:].values
X_mm = mms.fit_transform(X)
X_completo = np.concatenate((onehotlabels, X_mm), axis = 1)

print('Total de muestras:',len(dataset))
print('Valores categoricos en state:', dataset['State'].unique())
title = " Visualizacion de dataset "
print(f'{title:=^70}')
print(dataset.head())

modelo = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X_completo,y,test_size=0.1,random_state=0)
modelo.fit(X_train, y_train)
score_train = modelo.score(X_train, y_train)
score_test = modelo.score(X_test, y_test)
print('training score = {:7.3f}'.format(score_train))
print('test score = {:7.3f}'.format(score_test))

print("Intercepción = ",modelo.intercept_)
print("Coeficientes = ",modelo.coef_)