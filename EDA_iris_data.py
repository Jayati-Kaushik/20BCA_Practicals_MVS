# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 19:30:06 2022

@author: prajw
"""

import pandas as pd 
import numpy as np
import os 
import seaborn as sns
import matplotlib.pyplot as plt
os.chdir("C:/Users/prajw/OneDrive/Desktop/New folder")
iris = pd.read_csv('Iris.csv')
iris.head()
print(iris.head())
print(iris.describe())
sns.countplot(x ='Species', data = iris)
plt.show()
sns.scatterplot('SepalLengthCm','SepalWidthCm', hue= 'Species', data = iris)
plt.show()
sns.pairplot(iris.drop(['Id'],axis =1),hue= 'Species', height=2)
plt.show()
sns.boxenplot()
plt.show()
sns.heatmap(iris.corr(), data = iris)
plt.show()
x = iris.corr(method= 'pearson')
print(x)