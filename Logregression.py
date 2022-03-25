# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 23:53:27 2022

@author: SharwinA
"""

import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np
os.chdir('C:/Users/admin/Desktop/project')
df= pd.read_csv('iris.csv')
df.info()
df.describe()
df.head()
df.isnull()
df.count()
df.isna().sum()

sns.boxplot(x='sepal_length', y='sepal_width', data= df)
sns.barplot(x='sepal_length', y='sepal_width', data= df)
plt.scatter(x='sepal_length', y='sepal_width', data= df)
sns.pairplot(df)

#splitting Dataset
X = df.iloc[:, [0,1,2,3]].values
y = df.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


Logmodel= LogisticRegression(random_state = 0, solver='liblinear',multi_class= 'ovr')
Logmodel.fit(X_train, y_train)


y_pred= Logmodel.predict(X_test)
probs_y=Logmodel.predict_proba(X_test)
### Print results 
probs_y = np.round(probs_y, 2)
res = "{:<10} | {:<10} | {:<10} | {:<13} | {:<5}".format("y_test", "y_pred", "Setosa(%)", "versicolor(%)", "virginica(%)\n")
res += "-"*65+"\n"
res += "\n".join("{:<10} | {:<10} | {:<10} | {:<13} | {:<10}".format(x, y, a, b, c) for x, y, a, b, c in zip(y_test, y_pred, probs_y[:,0], probs_y[:,1], probs_y[:,2]))
res += "\n"+"-"*65+"\n"
print(res)
