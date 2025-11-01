# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 13:27:26 2021

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
 
np.random.seed(0)

x = np.linspace(-1, 1, 50)
y0 = np.sin(x*np.pi)

y = y0 + np.random.randn(len(x))*0.2

plt.scatter(x, y0, color='red')
plt.show()
plt.scatter(x, y, color= 'blue')
plt.show()

#################  Estimación con 1D ######## 
''' x debe ser 2D'''
x = x.reshape((-1,1))  
model1 = LinearRegression()
model1.fit(x, y)

y_pred = model1.predict(x)
score = model1.score(x,y)

" y = a*x +b"
a1=model1.coef_
b1=model1.intercept_

y_pred2 = a1[0]*x + b1
plt.scatter(x, y, color= 'blue')
plt.plot(x,y_pred,color='green')
#plt.plot(x,y_pred2, color='cyan')
plt.title(" Degree =1 : Score={:.3f}".format(score))
plt.show()

############## Estimación con polinomio de 2nd orden #########

degree=4  ### Puede modificar el valor de degree para observar "overfitting"
poly = PolynomialFeatures(degree=degree)
x_poly2 = poly.fit_transform(x)

model2 = LinearRegression()
model2.fit(x_poly2, y)

a2=model2.coef_
b2=model2.intercept_

y_pred = model2.predict(x_poly2)
score = model2.score(x_poly2,y)

plt.scatter(x, y, color= 'blue')
plt.plot(x,y_pred,color='green')
plt.title(" Degree = " + str(degree) + " : Score={:.3f}".format(score))
plt.show()



