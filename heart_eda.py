# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 19:19:34 2022

@author: tejas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
mydata=pd.read_csv("heart.csv")
print(mydata.shape)
print(mydata.head(10))
mydata.info()
print(mydata.describe())
df_desc=pd.DataFrame(mydata.describe())
df_desc.to_csv("/Users/tejas/PycharmProjects/stats_eda/describe.csv")
# Do we have duplicates?
print('Number of Duplicates:', len(mydata[mydata.duplicated()]))
# Do we have missing values?
print('Number of Missing Values:', mydata.isnull().sum().sum())
#sice fastingbp ang heartdisease are categorical values we convert that to str data type
mydata['FastingBS'] = mydata['FastingBS'].astype(str)
mydata['HeartDisease'] = mydata['HeartDisease'].astype(str)
print(mydata.dtypes)
corr_ht=mydata.corr()
print(corr_ht)
sns.heatmap(corr_ht)
plt.show()
print(mydata.nunique())
print(mydata['ChestPainType'].unique())
print(mydata['ST_Slope'].unique())
print(mydata['RestingECG'].unique())
sns.pairplot(mydata)
sns.relplot(x="Oldpeak",y="RestingBP",hue="Sex",data=mydata)
sns.catplot(x="RestingBP",kind='box',data=mydata)
plt.show()
#linear regression
x=mydata[['Age']]
y=mydata['RestingBP']
np.reshape(-1,1)
print(x.head(10))
#X_train = X[:-20]
#X_test = X[-20:]
#Y_train = Y[:-20]
#Y_test = Y[-20:]
X_train, X_test,Y_train,Y_test = train_test_split(x,y,test_size =0.2)
print(X_train)
reg = linear_model.LinearRegression()
print(reg.fit(X_train,Y_train))
print(reg.coef_)
A=r2_score(X,Y)
print(A)
B=mean_squared_error(X,Y)
print(B)

#multiple linear regression
X=mydata[['Age']]
Y=mydata[['RestingBP','Oldpeak']]
#X_train, X_test,Y_train,Y_test = train_test_split(x,y,test_size =0.2)
X_train = X[:-20]
X_test = X[-20:]
Y_train = Y[:-20]
Y_test = Y[-20:]

reg = linear_model.LinearRegression()
reg.fit(X_train,Y_train)
print(reg.coef_)
P=r2_score(X,Y)
print(P)
Q=mean_squared_error(X,Y)
print(Q)