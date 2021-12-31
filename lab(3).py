# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 08:29:16 2021

@author: admin
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

titanicc = pd.read_csv('titanicc.csv')
print(titanicc.head())
print(titanicc.describe())
sns.countplot(x =  'Survived', data= titanicc)
sns.scatterplot('Pclass','Age',hue='Fare', data= titanicc)
sns.pairplot(titanicc.drop(['PassengerId'], axis=1) , hue='Fare',height=2)
#sns.heatmap( corr_matrix() , data= titanic)
X = titanicc.corr(method='pearson')
print(X)
sns.heatmap(titanicc.corr(method='pearson').drop(['PassengerId'],axis=1).drop(['PassengerId'],axis=0))
plt.boxplot('Fare', data= titanicc)
plt.show()