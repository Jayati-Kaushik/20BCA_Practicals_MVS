# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 10:47:51 2021

@author: Deepak SK
"""

import pandas as pd 
import numpy as np
import os 
import seaborn as sns
import matplotlib.pyplot as plt
os.chdir("C:/Users/Deepak SK/Desktop/3rd sem/datasets")
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