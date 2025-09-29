# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 07:25:20 2021

@author: user
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import  OneHotEncoder 
from sklearn.preprocessing import LabelBinarizer

# importar el data set
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4].values


# determine categorical and numerical features
numerical = X.select_dtypes(include=['int64', 'float64'])
categorical = X.select_dtypes(include=['object', 'bool'])

enc = OneHotEncoder()
onehotlabels = enc.fit_transform(categorical).toarray()

# lb = LabelBinarizer()
# onehotlabels = lb.fit_transform(categorical)
onehotlabels = onehotlabels[:,1:]  # Evitando co-liealidad

X = numerical.iloc[:,:].values
##### Concatenar onehotlabels con X 
X_completo = np.concatenate((onehotlabels,X),axis=1)