# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 17:40:51 2021

@author: admin
"""


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data1=pd.read_csv('C:/Users/admin/Downloads/CARS.csv')
data1=data1.drop(columns=['CarName','symboling','carheight','stroke','compressionratio','peakrpm','citympg','highwaympg'])
print(data1.head())
print(data1.describe())
sns.countplot(x='enginetype', data=data1)
plt.show()
sns.scatterplot(x='enginetype',y ='price',data =data1)
plt.show()
sns.scatterplot(x='carlength',y='price',data=data1)
plt.show()
sns.scatterplot(x='carwidth',y='price',data=data1)
plt.show()
sns.pairplot(data1.drop(['car_ID'],axis=1),height=3)
plt.show()
x=data1.corr(method='pearson')
print(x)
sns.heatmap(data1.corr(method='pearson').drop(['car_ID'],axis=1).drop(['car_ID'],axis=0))
plt.show()
sns.boxplot(y="price",data=data1)
plt.show()
sns.boxplot(x='enginetype',y='price',data=data1)
plt.show()

X= data1[['wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower','carlength','carwidth']]
Y= data1['price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)
reg = LinearRegression()
reg.fit(X_train,Y_train)
Y_pred = reg.predict(X_test)
# The coefficients
print("Coefficients: \n", reg.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(Y_test, Y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(Y_test, Y_pred))
# Variance score
print("Variance score: {}".format(reg.score(X_test,Y_test))) 
