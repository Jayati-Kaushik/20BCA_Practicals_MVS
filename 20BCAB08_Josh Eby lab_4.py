# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 10:26:11 2021

@author: Josh
"""

#EDA+linear regression
import pandas as pd
import os
import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn import linear_model
#from sklearn.linear_model import LinearRegression


os.chdir("D:/NOTES/Datasets")
cars = pd.read_csv('CarPrice_Assignment.csv')
print(cars.head())
print(cars.describe())
sns.pairplot(cars)
sns.scatterplot(x='horsepower', y='price', data=cars)
sns.scatterplot(x='compressionratio', y='price', data=cars)
sns.scatterplot(x='enginesize', y='price', data=cars)
sns.scatterplot(x='cylindernumber', y='price', data=cars)
sns.countplot(x='enginetype', data=cars)
sns.scatterplot(x='enginetype', y='price', data=cars)
sns.scatterplot(x='carheight', y='price', data=cars)
sns.scatterplot(x='carwidth', y='price', data=cars)
sns.scatterplot(x='carlength', y='price', data=cars)
sns.scatterplot(x='wheelbase', y='price', data=cars)
sns.scatterplot(x='fueltype', y='price', data=cars)
#sns.pairplot(cars.drop(['car_ID'],axis=1),height=3)
sns.boxplot(y='price', data=cars)
sns.boxplot(x='enginetype', y='price', data=cars)
sns.boxplot(x='fueltype', y='compressionratio',data=cars)
#plt.show()

#one variable regression
x=cars[['price']]
y=cars[['highwaympg']]
reg=linear_model.LinearRegression()
#reg=LinearRegression()
#reg.fit([[0,0],[1,1],[2,2],[0,1,2]])
reg.fit(x,y)
print(reg.coef_)
sns.regplot(x,y)

#multiple  regression
X=cars[['horsepower','curbweight']]
Y=cars[['price']]
reg=linear_model.LinearRegression()
reg.fit(X,Y)
print(reg.coef_) 
