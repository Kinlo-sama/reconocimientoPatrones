# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 08:19:04 2021

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap

moons = make_moons(n_samples=500, noise=0.3, random_state=0)

X = moons[0]
Y = moons[1]
plt.figure(figsize=(12, 8))

for i, j in enumerate(np.unique(Y)):
    plt.scatter(X[Y == j, 0], X[Y== j, 1], s=80,
                c = ListedColormap(('red', 'green'))(i), label = j)

plt.legend()
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, random_state=0)


### Uso de Pipeline con el método de Kernel ####

kernel_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', gamma=5))
])

kernel_svm.fit(X_train, Y_train)

print(kernel_svm.score(X_test, Y_test))

X_set, y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, 
                               step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, 
                               step = 0.01))

plt.contourf(X1, X2, kernel_svm.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.4, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(Y)):
    plt.scatter(X[Y == j, 0], X[Y== j, 1], s=20,
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Test Data ')
plt.legend()
plt.show()



