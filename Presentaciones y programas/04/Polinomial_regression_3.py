# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 13:27:26 2021

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt

 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
 
np.random.seed(0)

x1 = np.linspace(-5, 5, 20)
x2 = np.linspace(-3, 3, 20)
x3 = np.linspace(-2, 2, 20)
x=np.vstack((x1,x2,x3)).T  #### generar una matriz 20 x 2(primer columna: x1, segundo columna x2)

y0 =  (-1) * x1 + 3 * (x1 ** 2) + 4 * x2 + 2 * (x2 ** 2) + 2 * (x3**2)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(x1,x2,y0, color='red')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.show()


#################  Estimación con 1D ######## 
''' x debe ser 2D'''
  
model1 = LinearRegression()
model1.fit(x, y0)

y_pred1 = model1.predict(x)
score1 = model1.score(x,y0)

" y = a*x +b"
a1=model1.coef_
b1=model1.intercept_

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(x1,x2,y0, color='red')
ax.plot(x1,x2,y_pred1, color='green')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title("Degree = 1 : Score={:.3f}".format(score1))
plt.show()



############## Estimación con polinomio de 2nd orden #########

degree=2
poly = PolynomialFeatures(degree=degree)
x_poly2 = poly.fit_transform(x)

model2 = LinearRegression()
model2.fit(x_poly2, y0)

a2=model2.coef_
b2=model2.intercept_

y_pred2 = model2.predict(x_poly2)
score2 = model2.score(x_poly2,y0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(x1,x2,y0, color='red')
ax.plot(x1,x2,y_pred2, color='green')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title("Degree = "+ str(degree) +" : Score={:.3f}".format(score2))
plt.show()


############## Estimación con polinomio de 3er orden ########### 
degree=3
poly = PolynomialFeatures(degree=degree)
x_poly3 = poly.fit_transform(x)

model3 = LinearRegression()
model3.fit(x_poly3, y0)

a3=model3.coef_
b3=model3.intercept_

y_pred3 = model3.predict(x_poly3)
score3 = model3.score(x_poly3,y0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(x1,x2,y0, color='red')
ax.plot(x1,x2,y_pred3, color='green')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title("Degree = "+ str(degree) +" : Score={:.3f}".format(score3))
plt.show()


