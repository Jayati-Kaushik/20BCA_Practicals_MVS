# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 17:03:16 2021

@author: joelpaul
"""

import numpy as np
x=np.array([1,2,3,4,5,6,7])
print(x)
print(x[1],x[3³],x[6])
print(np.shape(x))
y=x.T
print(y)
a=np.array([[1,2,3,4,5,],[5,6,7,8,9,],[9,8,7,6,5],[1,3,5,7,9]]) print(a[0,0], a[0,1],a[0,2],a[0,3], a[0,4])
print(a)
print(np.shape(a))
b=a. T
print (b)
print(np.shape(b))
print(a[0,0], a[1,0],a[2,0],a[3,0])
k=np.zeros((7))
print (k)
s=np.ones ((3))
print(s)
 



import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir("E:/Desktop/college/Python")
iris = pd.read_csv('Iris.csv')
print(iris.head())
print(iris.describe())
sns.countplot(x='Species', data = iris)
plt.show()
sns.scatterplot('SepalLengthCm','SepalWidthCm', hue='Species',data = iris)
plt.show()
sns.pairplot(iris.drop(['Id'],axis =1), hue= 'Species', height= 2)
plt.show()
sns.heatmap(iris.corr(), data = iris)
plt.show()
x= iris.corr(method= 'pearson')
print(x)
sns.heatmap(iris.corr(method='pearson').drop(['Id'],axis=1).drop(['Id'],axis=0))



import numpy as np
X=np.array([[1,2,3],[3,2,1],[9,10,7]])
print(X)
Y=np.array([[1,2,3],[3,2,1],[9,10,7]]).T
print(X)
A=np.array([[0]*3])
print(A)
print(np.zeros((4,8)))
print(np.ones(4))
print(X)
print(Y)
print(X+Y)
print(np.shape(X))
I=np.array([[10,20,30,40,50,60,70]])
print(I)
print(I[0,1],I[0,3],I[0,6])
print(np.shape(I))
print(I.T)
V=np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]])
print(V)
print(np.shape(V))
print(V.T)
print(V[:,0])
print(V[0,:])
print(np.zeros(7))
print(np.ones(3))



import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir("E:/Desktop/college/Python")
iris = pd.read_csv('Iris.csv')
print(iris.head())
print(iris.describe())
sns.countplot(x='Species', data = iris)
plt.show()
sns.scatterplot('SepalLengthCm','SepalWidthCm', hue='Species',data = iris)
plt.show()
sns.pairplot(iris.drop(['Id'],axis =1), hue= 'Species', height= 2)
plt.show()
sns.heatmap(iris.corr(), data = iris)
plt.show()
x= iris.corr(method= 'pearson')
print(x)
sns.heatmap(iris.corr(method='pearson').drop(['Id'],axis=1).drop(['Id'],axis=0))