# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 18:49:44 2022

@author: KAIF
"""

import numpy as np
import pandas as pd
import seaborn as sns

data = pd.read_csv("C:/Users/ADMIN/Documents/COVID-19.csv")

print(data.shape)
print(data.head)
print(data.tail)
print(data.describe)
print(data.columns)
print(data.info())

print(data.nunique())

print(data['province'].unique())

#checking for null values
print(data.isnull().sum())

print(data.drop(['latitude','longitude'], axis=1))
print(data.head())

print(data.corr())
sns.heatmap(data.corr())
sns.pairplot(data)
sns.relplot(x="country",y="confirmed_cases",hue="recovered",data=data)
sns.distplot(data['country'])
sns.catplot(x='country', kind='box',data=data)

#multiple linear regression
x=data[['country']]
Y=data[['confirmed_cases','recovered']]
X_train= X[:-20]
X_test=X[-20:]
Y_train=Y[:-20]
Y_test=Y[-20:]

reg=linear_model.LinearRegression()
reg.fit(X_train,Y_train)
print(reg.coef_)
P=r2_score(X,Y)
print(P)











