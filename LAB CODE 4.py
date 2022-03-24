# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 19:35:11 2022

@author: musai
"""
#EDA and linear regression , ,  

 
 

import pandas as pd 

import os 

import seaborn as sns 

#import matplotlib.pyplot as plt 

from sklearn import linear_model 

#from sklearn.linear_model import LinearRegression 

 
 

os.chdir("C:\Users\musai\OneDrive\Desktop\SEMESTER 3\DATA MINING SEM 3")

mtcars = pd.read_csv('CarPrice_Assignment.csv') 

print(mtcars.head()) 

print(mtcars.describe()) 

sns.pairplot(mtcars) 

sns.scatterplot(x='horsepower', y='price', data=mtcars) 

sns.scatterplot(x='compressionratio', y='price', data=mtcars) 

sns.scatterplot(x='enginesize', y='price', data=mtcars) 

sns.scatterplot(x='cylindernumber', y='price', data=mtcars) 

sns.countplot(x='enginetype', data=mtcars) 

sns.scatterplot(x='enginetype', y='price', data=mtcars) 

sns.scatterplot(x='carheight', y='price', data=mtcars) 

sns.scatterplot(x='carwidth', y='price', data=mtcars) 

sns.scatterplot(x='carlength', y='price', data=mtcars) 

sns.scatterplot(x='wheelbase', y='price', data=mtcars) 

sns.scatterplot(x='fueltype', y='price', data=mtcars) 

#sns.pairplot(mtcars.drop(['car_ID'],axis=1),height=3) 

sns.boxplot(y='price', data=mtcars) 

sns.boxplot(x='enginetype', y='price', data=mtcars) 

sns.boxplot(x='fueltype', y='compressionratio',data=mtcars) 

#plt.show() 

 
 

#one variable regression 

x=mtcars[['price']] 

y=mtcars[['highwaympg']] 

reg=linear_model.LinearRegression() 

#reg=LinearRegression() 

#reg.fit([[0,0],[1,1],[2,2],[0,1,2]]) 

reg.fit(x,y) 

print(reg.coef_) 

sns.regplot(x,y) 

 
 

#multiple regression 

X=mtcars[['horsepower','curbweight']] 

Y=mtcars[['price']] 

reg=linear_model.LinearRegression() 

reg.fit(X,Y) 

print(reg.coef_) 
