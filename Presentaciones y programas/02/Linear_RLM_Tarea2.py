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
dataset = pd.read_csv('fish.csv')

"""
1. Usando Length1, Length2, Length3, Height y Width Predecir Weight

2. Usando Species, Length1, Length2, Length3, Height y Width Predecir Weight

Analizar Specie de pez influye o no a la predicción de peso de pez.

'null' --> 0    promedio de especie, 

"""