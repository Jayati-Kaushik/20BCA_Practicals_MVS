from sklearn import linear_model
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
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
#To check the missing values
pd.isnull(cars)
#There is no missing values in the dataset
#To calculate the correlation
correlation = cars.corr()
print(correlation)
print(correlation['price'])
#Using countplot
sns.countplot(x="enginetype",data= cars)
plt.show()
#using pairplot for EDA
sns.pairplot(cars.drop(["car_ID"],axis=1),height=3)
plt.show()
sns.boxplot(y="price",data=cars)
plt.show()
#performing multiple linear regression
sns.barplot(x=cars.cylindernumber,y=cars.price,ci=None)

