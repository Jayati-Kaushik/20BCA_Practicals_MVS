from sklearn import linear_model
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#For creating a directory
import os
#Returning the current working directory
cwd = os.getcwd()
#Changing the current working directory
os.chdir('C:/Users/SONY/Desktop/car data')
cars=pd.read_csv('cars.csv')
print(cars)
#Exploratory data analysis
print(cars.describe())
#Returns the first five rows
print(cars.head())
#Returns the last five rows
print(cars.tail())
#To check the missing values
pd.isnull(cars)
#There is no missing values in the dataset
#To calculate the correlation
correlation = cars.corr()
print(correlation)
print(correlation['price'])
#Simple scatter plot
sns.scatterplot(x='horsepower',y='price',hue='CarName',data=cars)
plt.title('Horsepower vs Price')
plt.xlabel('horsepower')
plt.ylabel('price')
#Performing simple linear regression
X = cars['horsepower'].values.reshape(-1,1)
y = cars['price'].values.reshape(-1,1)
#Here the test_size variable actually species the proportion of the test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#Training the algorithm
regressor = LinearRegression()  
regressor.fit(X_train, y_train)
#To retrieve the intercept:
print(regressor.intercept_)
#The intercept value is negative.This means that the expected value of our price will be less than 0 when all horsepower values are set to 0.
#For retrieving the slope:
print(regressor.coef_)
plt.scatter(X_train, y_train, color = "red")
#Plotting a simple linear regression
plt.plot(X_train, regressor.predict(X_train), color = "green")
plt.title("Horsepower vs price")
plt.xlabel("horsepower")
plt.ylabel("price")
plt.show()





