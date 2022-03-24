# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 19:34:15 2022

@author: musai
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
import seaborn as sns
iris=datasets.load_iris()
iris=pd.DataFrame(data=iris.data, columns= iris.feature_names)
iris.head(8)
print(iris.head(5))
print(iris.tail(3))
print(iris.describe())
iris.isnull()
plt.plot(iris)
iris.plot(kind= "scatter", x=0, y=1)
iris.plot(kind="bar", x=0,y=1)
sns.set_style("whitegrid")
print(iris)
sns.pairplot(iris)
iris.info()
iris['sepal length (cm)'].hist()
iris['sepal length (cm)'].hist()
print(iris.corr)