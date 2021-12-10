# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 18:11:43 2021
@author: Admin
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

best = pd.read_csv('bestsellers.csv')
print(best.head())
print(best.describe())
sns.countplot(x =  'User Rating', data= best)
plt.show()
sns.scatterplot(x='Price' ,y='Reviews', data= best)
plt.show()
sns.scatterplot(x='Genre',y='Reviews', data= best)
plt.show()
sns.pairplot(best.drop(['Name'], axis=1) ,height=2)
plt.show()
A = best.corr(method='pearson')
print(A)
sns.heatmap(best.corr(method='pearson').drop(['Year'],axis=1).drop(['Year'],axis=0))
plt.show()
plt.boxplot('User Rating', data= best)
plt.show()

#regression for one variable
X= best[['User Rating']]
Y = best['Price']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)
reg = linear_model.LinearRegression()
reg.fit(X_train,Y_train)
Y_pred = reg.predict(X_test)
print("Coefficients: \n", reg.coef_)
print("Mean squared error: %.2f" % mean_squared_error(Y_test, Y_pred))
print("Coefficient of determination: %.2f" % r2_score(Y_test, Y_pred))
print("Variance score: {}".format(reg.score(X_test,Y_test)))
sns.regplot(X,Y)

#Multiple regrssion
X = best[['User Rating','Reviews','Year']]
Y = best['Price']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)
reg = linear_model.LinearRegression()
reg.fit(X_train,Y_train)
Y_pred = reg.predict(X_test)
print("Coefficients: \n", reg.coef_)
print("Mean squared error: %.2f" % mean_squared_error(Y_test, Y_pred))
print("Coefficient of determination: %.2f" % r2_score(Y_test, Y_pred))
print("Variance score: {}".format(reg.score(X_test,Y_test))) 