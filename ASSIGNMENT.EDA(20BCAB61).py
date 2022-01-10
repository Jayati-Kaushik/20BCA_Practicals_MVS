# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 22:30:51 2022

@author: ADMIN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
mydata=pd.read_csv("C:/Users/ADMIN/Downloads/medals.csv")
print(mydata.shape)
print(mydata.head(10))
print(mydata.tail(10))

print(mydata.describe())
print(mydata.columns)
print(mydata.info)
print('Number of missing values:', mydata.isnull().sum())

print(mydata.corr())
sns.heatmap(mydata.corr())
sns.pairplot(mydata)
sns.relplot(x='Year',y='City',hue='Medal',data=mydata)
sns.displot(mydata['Sport'])
sns.catplot(x='Year',kind='box',data=mydata)