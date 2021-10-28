# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 19:25:37 2021

@author: Josh
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pwd
iris = pd.read_csv('Iris.csv')
print(iris.head())
print(iris.describe())
sns.countplot(x='Species', data=iris)
sns.scatterplot('SepalLengthCm','SepalWidthCm', hue='Species',data=iris)
sns.pairplot(iris.drop(['Id'], axis =1),hue='Species',height=2)
#sns.boxenplot()
x=iris.corr(method='pearson')
print(x)
sns.heatmap(iris.corr_matrix(),method='pearson'.drop(['Id'],axis=1).drop(['Id'],axis=0))
plt.boxplot('SepalWidthCm', data=iris)

plt.show()
