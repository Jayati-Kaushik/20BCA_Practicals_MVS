# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 18:54:29 2021

@author: vlaks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates
from datetime import datetime
import sklearn
import seaborn as sns
import os
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

os.chdir("C:/Users/Aarushi/Desktop/R Studio")
wdata = pd.read_csv("Store_sales.csv")
print(wdata.head())
print(wdata.dtypes)
print(wdata.info())
wdata['Date'] =  pd.to_datetime(wdata['Date'])
print(wdata.info())

#Finding missing values
print(wdata.isnull().sum())

#Finding mean, standard deviation
print(wdata.describe())

#splitting date into day, month and year
wdata["Day"]= pd.DatetimeIndex(wdata['Date']).day
wdata['Month'] = pd.DatetimeIndex(wdata['Date']).month
wdata['Year'] = pd.DatetimeIndex(wdata['Date']).year
print(wdata.head())
print(wdata.describe())

sns.countplot(x='Weekly_Sales', data = wdata)
print(plt.show())
sns.scatterplot('Month','Temperature',data = wdata)
print(plt.show())
sns.lmplot(x ='Day', y ='Weekly_Sales', data = wdata, order = 2, ci = None)
sns.lmplot(x ='Unemployment', y ='Weekly_Sales', data = wdata, order = 2, ci = None)
X = wdata[['Temperature']]
Y = wdata['Weekly_Sales']
print(X.head(10))
X = wdata[['Temperature','Fuel_Price']] #for multivariate regression

#splitting data into test and training set
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size =0.2)
print(x_train)

random_state = 10
clf = LinearRegression()
#fitting x_train and y_train variables
print(clf.fit(x_train,y_train))
print(clf.coef_)
#print(plt.scatter(x_train, y_train))
print(sns.regplot(x_train, y_train))


clf.predict(x_test)
clf.score(x_test,y_test)
