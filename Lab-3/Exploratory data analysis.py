import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#For creating a directory
import os
#Returning the current working directory
cwd = os.getcwd()
#Changing the current working directory
os.chdir('C:/Users/SONY/Desktop/car data')
cars=pd.read_csv('cars.csv')
print(cars)
#Returns the first six rows
print(cars.head())
print(cars.describe())
sns.countplot(x='CarName',data=cars)
plt.show()
sns.scatterplot(x='carlength',y='carwidth',hue='CarName',data=cars)
plt.show()
X=cars.corr(method='pearson')
print(X)
sns.heatmap(cars.corr(method='pearson').drop(['car_ID'],axis=1).drop(['car_ID'],axis=0))
plt.show()
