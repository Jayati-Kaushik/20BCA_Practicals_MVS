# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 20:14:20 2022

@author: asus
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
os.chdir("C:/Users/S Hema/Desktop/R/r")
gdata = pd.read_csv("Groceries_dataset.csv")
#EDA
print(gdata.head())
print(gdata.dtypes)
print(gdata.info())
gdata['Date'] =  pd.to_datetime(gdata['Date'])
print(gdata.info())
print(gdata.isnull().sum())
print(gdata.describe())
#multivariate regression
sns.scatterplot('Member_number','itemDescription',data = gdata)
print(plt.show())
X = gdata[['Member_number']]
Y = gdata['itemDescription']
print(X.head(10))
X = gdata[['Member_number','itemDescription']]
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size =0.3)
print(x_train) 
#linear regression 
random_state = 20
l = LinearRegression()
print(l.fit(x_train,y_train))
print(l.coef_)
print(sns.regplot(x_train, y_train)
l.predict(x_test)
l.score(x_test,y_test)