# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 11:24:48 2021

@author: Rose Maria Thomas
"""
#EDA
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir("C:/Users/Rose Maria Thomas/Desktop/joel")
data=pd.read_csv('insurance.csv')
print(data.head())
print(data.describe())
sns.countplot(x='charges', data = data)
plt.show()
sns.scatterplot('bmi','children', hue='charges',data = data)
plt.show()
sns.pairplot(data.drop(['age'],axis =1), hue= 'charges', height= 2)
plt.show()
sns.boxenplot(x='bmi',y='charges',hue='children',data=data)
plt.show()
sns.heatmap(data.corr(), data =data)
plt.show()
x= data.corr(method= 'pearson')
print(x)
