import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
os.chdir("C:/Users/Seizal Pathania/Downloads/Rfiles")
iris = pd.read_csv('iris_csv.csv')
iris.head()
print(iris.head())
print(iris.describe())
sns.countplot(x='class', data = iris)
plt.show()
sns.scatterplot('sepallength','sepalwidth', hue='class',data = iris)
plt.show()
sns.pairplot(iris.drop(['id'],axis =1), hue= 'class', height= 2)
plt.show()
sns.boxenplot()
plt.show()
sns.heatmap(iris.corr(), data = iris)
plt.show()
x= iris.corr(method= 'pearson')
print(x)