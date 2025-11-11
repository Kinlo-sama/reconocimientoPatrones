# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 19:52:25 2020

KNN_3.py

@author: Mariko Nakano
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

### Datos tienen 30 dimensiones #####
"""
   solo dibujar relación entre los primeros dos características
"""

X_1_blue = cancer.data[:,0][cancer.target ==0]
X_2_blue = cancer.data[:,2][cancer.target ==0]

X_1_red = cancer.data[:,0][cancer.target ==1]
X_2_red = cancer.data[:,2][cancer.target ==1]

plt.scatter(X_1_blue,X_2_blue, c='blue',alpha=0.7, label="benigno")
plt.scatter(X_1_red,X_2_red,c='red',alpha=0.7, label="maligno")
plt.title("Brest_cancer_dataset")
plt.xlabel('mean_radius')
plt.ylabel('mean perimeter')
plt.legend()
plt.show()


X=cancer.data
y=cancer.target

#df1 = pd.DataFrame(X)
#sn.pairplot(df1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.75, 
                                                    stratify=cancer.target, random_state=1)
from sklearn.neighbors import KNeighborsClassifier
# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix

num_vecinos = [1,3,5,7,9,11]
accuracy=[]
for i in num_vecinos:
    clf = KNeighborsClassifier(n_neighbors=i, weights="distance")
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    acc=clf.score(X_test,y_test)
    print('accuracy={}'.format(acc))
    accuracy.append(acc)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)


### usar Grid_Search


 