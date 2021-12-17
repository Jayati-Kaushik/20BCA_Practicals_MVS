# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 11:43:06 2021

@author: dell
"""

import pandas as pd
import seaborn as sbn
import numpy as np
import matplotlib.pyplot as plt
data1 = pd.read_csv(C:\Users\dell\.spyder-py)
print(data1)
data1.head()
print(data1.head())
# for data redundancy
print(data1.nunique())
print(data1.isnull().sum())
data = data1.drop(['bmi','region'],axis=1)
print(data)
correlation = data.corr()
#correlaion matrix
print(sbn.heatmap(correlation))
print(sbn.pairplot(data))
print(sbn.relplot(x = 'age',y = 'children',hue = 'sex',data= data))
print(sbn.distplot(data['children']))
print(sbn.catplot(x = 'children',kind='box',data= data))