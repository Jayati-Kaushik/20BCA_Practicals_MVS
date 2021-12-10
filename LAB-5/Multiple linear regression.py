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

#performing linear regression
X = cars['horsepower'].values.reshape(-1,1)
y = cars['price'].values
ols = linear_model.LinearRegression()
model = ols.fit(X, y)
response = model.predict(X)
plt.plot(X, response, color='k', label='Regression model')
plt.scatter(X, y, edgecolor='k', facecolor='red', label='Car data')

#Coefficient of determination
r2 = model.score(X, y)
print('r2 score for perfect model is', r2)
#Therefore 65% of the data fits the model

#performing multiple linear regression
X=cars[['horsepower','curbweight']]
Y=cars[['price']]
reg=linear_model.LinearRegression()
reg.fit(X,Y)
print(reg.coef_)

cars = pd.DataFrame(cars,columns=['horsepower','curbweight','price'])
#This module is for multiple regression
import statsmodels.formula.api as smf
model = smf.ols(formula='price ~ horsepower + curbweight', data=cars)
results_formula = model.fit()
results_formula.params

x_pred, y_pred = np.meshgrid(np.linspace(cars.horsepower.min(), cars.horsepower.max(), 100),np.linspace(cars.curbweight.min(), cars.curbweight.max(), 100))
X= pd.DataFrame({'horsepower': x_pred.ravel(), 'curbweight': y_pred.ravel()})
fittedY=results_formula.predict(exog=X)

# converting the predicted result into array
fittedY=np.array(fittedY)

#Visualization for multiple linear regression
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(cars['horsepower'],cars['curbweight'],cars['price'],c='red', marker='o', alpha=0.5)
ax.plot_surface(x_pred,y_pred,fittedY.reshape(x_pred.shape), color='b', alpha=0.3)
ax.set_xlabel('Horsepower')
ax.set_ylabel('Curbweight')
ax.set_zlabel('Price')
plt.show()
