# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 20:06:46 2021

@author: Harshitha
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

iris = pd.read_csv('Iris.csv')
#print(iris.head())
#print(iris.desribe())
sns.countplot(x = 'species', data= iris)
sns.scatterplot('sepalLengthCm','sepalWidth','hue = species', data=iris )
sns.pairplot(iris.drop(['Id'], axis =1), hue = 'species', height=2)
sns.boxplot()
sns.heatmap(corr_matrix(), data = iris)
x=iris.corr(method='pearson')
print(x)
plt.show() 