# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 10:32:59 2021

@author: DEC BEAST
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

titanic = pd.read_csv('titanic.csv')
print(titanic.head())
print(titanic.describe())
sns.countplot(x =  'Survived', data= titanic)
sns.scatterplot('Pclass','Age',hue='Fare', data= titanic)
sns.pairplot(titanic.drop(['PassengerId'], axis=1) , hue='Fare',height=2)
#sns.heatmap( corr_matrix() , data= titanic)
X = titanic.corr(method='pearson')
print(X)
sns.heatmap(titanic.corr(method='pearson').drop(['PassengerId'],axis=1).drop(['PassengerId'],axis=0))
plt.boxplot('Fare', data= titanic)
plt.show()