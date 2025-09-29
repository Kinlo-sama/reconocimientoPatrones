# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 21:23:23 2020

Regresión lineal simple tarea

@author: Mariko Nakano
"""

from sklearn import datasets
import pandas as pd

wine_data = datasets.load_wine()
wine = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)

''' Obtener relación entre total_phenole y flavanoids 
Aplicar Regresión lineal de sklearn
  (1) obtener función de línea
  (2) dibujar la linea, junto con los datos
  (3) obtener Score

   Obtener relación entre hue y color_intensity
  (1) obtener función de línea
  (2) dibujar la linea, junto con los datos
  (3) obtener Score   
  
  Analizar relaciones de dos factores (total_phenole y flavenoids) y 
  (hue y color_intensity) 
  
  Seleccionar dos factores de caracteristicas de vinos que mejor relacionado 
  linealmente
  '''
  
  