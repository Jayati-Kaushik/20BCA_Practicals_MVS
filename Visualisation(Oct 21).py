# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 23:47:36 2021

@author: FIONA
"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
os.chdir("C:/Users/FIONA/Desktop/Stats Lab")
iris = pd.read_csv('Iris.csv')
iris.head()
print(iris.head())
print(iris.describe())
sns.countplot(x='class', data = iris)
plt.show()
sns.scatterplot('SepalLength','SepaLwidth', hue='class',data = iris)
plt.show()
sns.pairplot(iris.drop(['Id'],axis =1), hue= 'class', height= 2)
plt.show()
sns.boxenplot()
plt.show()
sns.heatmap(iris.corr(), data = iris)
plt.show()
x= iris.corr(method= 'pearson')
print(x)