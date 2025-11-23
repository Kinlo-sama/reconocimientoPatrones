# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 13:16:54 2021

@author: Mariko Nakano
Uso de SNS
"""
import seaborn as sns

df = sns.load_dataset("iris")
sns.set(font_scale=1.5)
sns.pairplot(df,hue = 'species', diag_kind='kde')


