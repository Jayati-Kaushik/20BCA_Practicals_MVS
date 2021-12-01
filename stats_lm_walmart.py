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

os.chdir("C:/Users/Seizal Pathania/Downloads/Rfiles")
wdata = pd.read_csv("Walmart_Store_sales.csv")
print(wdata.head())
print(wdata.dtypes)
print(wdata.info())

# Convert date to datetime format 
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

#sns.countplot(x='Weekly_Sales', data = wdata)
#print(plt.show())
sns.scatterplot('Month','Temperature',data = wdata)
print(plt.show())
#sns.lmplot(x ='Day', y ='Weekly_Sales', data = wdata, order = 2, ci = None)
#sns.lmplot(x ='Unemployment', y ='Weekly_Sales', data = wdata, order = 2, ci = None)
X = wdata[['Temperature']]
Y = wdata['Weekly_Sales']
print(X.head(10))


#splitting into test and training set
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size =0.2)
# print the data
print(x_train)

random_state = 10
#creating an object of the LinearRegression class.
clf = LinearRegression()
#fitting x_train and y_train variables
print(clf.fit(x_train,y_train))
print(clf.coef_)
#print(plt.scatter(x_train, y_train))
print(sns.regplot(x_train, y_train))

#predicting
clf.predict(x_test)
#returning coeff of determination of prediction
clf.score(x_test,y_test)