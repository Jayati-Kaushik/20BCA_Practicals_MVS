# -*- coding: utf-8 -*-
"""


@author: naveen
"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import datasets
import sklearn

os.chdir("D:/Downloads")
cars=pd.read_csv('CarPrice_Assignment.csv')
print(cars.describe())
cars.info()

#3. Regression on multiple variables
X=cars[['horsepower','curbweight']]
Y=cars[['price']]
reg=linear_model.LinearRegression()
reg.fit(X,Y)
print(reg.coef_)

# Lohith Ashwa explained me this code.
df2 = pd.DataFrame(cars,columns=['horsepower','curbweight','price'])
import statsmodels.formula.api as smf
model = smf.ols(formula='price ~ horsepower + curbweight', data=df2)
results_formula = model.fit()
results_formula.params

## Preparing the data for Visualization

x_surf, y_surf = np.meshgrid(np.linspace(df2.horsepower.min(), df2.horsepower.max(), 100),np.linspace(df2.curbweight.min(), df2.curbweight.max(), 100))
onlyX = pd.DataFrame({'horsepower': x_surf.ravel(), 'curbweight': y_surf.ravel()})
fittedY=results_formula.predict(exog=onlyX)

## converting the predicted result in an array
fittedY=np.array(fittedY)

# Visualize the Data for Multiple Linear Regression
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df2['horsepower'],df2['curbweight'],df2['price'],c='red', marker='o', alpha=0.5)
ax.plot_surface(x_surf,y_surf,fittedY.reshape(x_surf.shape), color='b', alpha=0.3)
ax.set_xlabel('Horsepower')
ax.set_ylabel('Curbweight')
ax.set_zlabel('Price')
plt.show()