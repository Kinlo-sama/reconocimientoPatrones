# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 14:34:18 2021

Brest Cancer 

@author: Mariko Nakano 
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cancer = load_breast_cancer()
feature_list=list(cancer.feature_names)

df=pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
df['tipo de cancer']=cancer.target
sub_df=df.iloc[:,:3].copy()
sub_df["tipo de cancer"]=cancer.target

name_list =[]
for i in range(len(df)):
    tipo = 'maligno'
    if sub_df.iloc[i,3] == 1:
        tipo ='benigno'
    name_list.append(tipo)

    
sub_df["tipo de cancer letra"]=name_list
sns.set(font_scale=1.3)
sns.pairplot(sub_df.drop(columns='tipo de cancer'), hue="tipo de cancer letra", 
             hue_order=['benigno','maligno'])
plt.show()

X=sub_df.iloc[:,:2].values
y=sub_df.iloc[:,3].values

kernel_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='poly', degree=8, coef0=1.5))
])


for i, j in enumerate(np.unique(y)):
    plt.scatter(X[y == j, 0], X[y== j, 1], s=10,
                color = ListedColormap(('red', 'green'))(i), 
                label = cancer.target_names[j])

plt.legend()
plt.xlabel(feature_list[0])
plt.ylabel(feature_list[1])
plt.show()

################# Clasificación #####
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, 
                                                    random_state=0)

### Uso de Pipeline con el método de Kernel ####

# kernel_svm = Pipeline([
#     ('scaler', StandardScaler()),
#     ('svm', SVC(kernel='sigmoid'))
# ])

kernel_svm.fit(X_train, y_train)

score = kernel_svm.score(X_test, y_test)

X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, 
                               step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, 
                               step = 0.01))

plt.contourf(X1, X2, kernel_svm.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.4, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y)):
    plt.scatter(X[y == j, 0], X[y== j, 1], s=10,
                color = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Test Data:  Accuracy='+str(np.round(score,2)))
plt.legend()
plt.show()
