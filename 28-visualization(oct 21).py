# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 11:10:57 2021

@author: naveen

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
sns.countplot(x='class', data = iris)
plt.show()
sns.scatterplot('SepalLength','SepalWidth', hue='class',data = iris)
plt.show()
sns.pairplot(iris.drop(['Id'],axis =1), hue= 'class', height= 2)
plt.show()
sns.boxenplot()
plt.show()
sns.heatmap(iris.corr(), data = iris)
plt.show()
x= iris.corr(method= 'pearson')
print(x)
