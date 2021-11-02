# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 23:08:16 2021

@author: chris
"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
os.chdir("C:/Users/chris/Desktop/Jk Python Prac")
iris = pd.read_csv('Iris.csv')
iris.head()
print(iris.head())
print(iris.describe())
sns.countplot(x='Species', data = iris)
plt.show()
sns.scatterplot('SepalLengthCm','SepalWidthCm', hue='Species',data = iris)
plt.show()
sns.pairplot(iris.drop(['Id'],axis =1), hue= 'Species', height= 2)
plt.show()
sns.boxenplot()
plt.show()
sns.heatmap(iris.corr(), data = iris)
plt.show()
x= iris.corr(method= 'pearson')
print(x)



