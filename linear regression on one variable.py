# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 15:15:09 2021

@author: ACER
"""
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import datasets
import sklearn

os.chdir("C:/Users/ACER/Downloads")
cars=pd.read_csv('CarPrice_Assignment.csv')
print(cars.describe())
cars.info()

# 1. EDA and visualisation 
sns.countplot('doornumber',data=cars)
plt.show()

plt.hist('cylindernumber',data = cars)
plt.show()

x= cars.corr(method= 'pearson')
print(x)

sns.heatmap(cars.corr(method='pearson').drop(['car_ID','symboling'],axis=1).drop(['car_ID','symboling'],axis=0),data=cars)
sns.show(heatmap)

df = pd.DataFrame(cars,columns=['cylindernumber','horsepower'])
plt.bar(df['cylindernumber'], df['horsepower'])
plt.title('Cylinder number vs Horsepower', fontsize=14)
plt.xlabel('CYlinder Number', fontsize=14)
plt.ylabel('Horse Power', fontsize=14)
plt.show()


sns.pairplot(cars)
plt.show()

sns.boxplot(y='compressionratio',x='fueltype',data=cars)
plt.show()


#Q1. Regression on one variable 
#1a Regression on one variable for negative correlation
X=cars[['highwaympg']]
Y=cars[['horsepower']]
reg=linear_model.LinearRegression()
reg.fit(X,Y)
print(reg.coef_)
sns.regplot(X,Y)
plt.show()

#1b Regression on one variable for positive correlation
X=cars[['wheelbase']]
Y=cars[['carlength']]
reg=linear_model.LinearRegression()
reg.fit(X,Y)
print(reg.coef_)
sns.regplot(X,Y)

#1c Regression on one variable with no correlation
X=cars[['stroke']]
Y=cars[['price']]
reg=linear_model.LinearRegression()
reg.fit(X,Y)
print(reg.coef_)
sns.regplot(X,Y)