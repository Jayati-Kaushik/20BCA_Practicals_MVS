import pandas as pd
import os
import seaborn as sns
from sklearn import linear_model 

os.chdir("C:/Users/Santosh/Desktop/Neha")
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

sns.boxplot(y='price', data=mtcars)
sns.boxplot(x='enginetype', y='price', data=mtcars)
sns.boxplot(x='fueltype', y='compressionratio',data=mtcars)

#linear regression
x=mtcars[['price']]
y=mtcars[['highwaympg']]
reg=linear_model.LinearRegression()
reg.fit(x,y)
print(reg.coef_)
sns.regplot(x,y)

#multiple  regression
X=mtcars[['horsepower','curbweight']]
Y=mtcars[['price']]
reg=linear_model.LinearRegression()
reg.fit(X,Y)
print(reg.coef_)