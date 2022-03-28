# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 11:44:53 2021

@author: Rose Maria Thomas
"""
import os
import pandas as pd
from sklearn import linear_model as lm 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn import datasets
import sklearn 


os.chdir("C:/Users/Rose Maria Thomas/Desktop/joel")
data=pd.read_csv('insurance.csv')
print(data.describe())
print(data.head()) 
print(data.dtypes)
print(data.info)
data['sex']=data['sex'].astype(str)
data['smoker']=data['smoker'].astype(str)
data['region']=data['region'].astype(str)
co=data.corr(method='pearson')
print(co)
sns.heatmap(co)
sns.pairplot(data)
plt.show()
x=data[['bmi']]
print(x)
y=data[['charges']]
print(y)
x_train=x[:-1000]
x_test=x[-1000:]
y_train=y[:-1000]
print(y_train)
print(x_train)
y_test=y[-1000:]

reg=lm.LinearRegression()
reg.fit(x_train,y_train)
print(reg.coef_)
sns.regplot(x,y)


a=data[['bmi','age']]
print(a)
b=data[['charges']]
print(b)
a_train=a[:-1000]
a_test=a[-1000:]
b_train=b[:-1000]
print(b_train)
print(a_train)
b_test=b[-1000:]

reg=lm.LinearRegression()
reg.fit(a_train,b_train)
print(reg.coef_)


df2 = pd.DataFrame(data,columns=['bmi','age','charges'])
import statsmodels.formula.api as smf
model = smf.ols(formula='bmi ~ age + charges', data=df2)
results_formula = model.fit()
results_formula.params


## Prepare the data for Visualization

x_surf, y_surf = np.meshgrid(np.linspace(df2.bmi.min(), df2.bmi.max(), 100),np.linspace(df2.age.min(), df2.charges.max(), 100))
onlyX = pd.DataFrame({'bmi': x_surf.ravel(), 'age': y_surf.ravel()})
fittedY=results_formula.predict(exog=onlyX)



## convert the predicted result in an array
fittedY=np.array(fittedY)




# Visualize the Data for Multiple Linear Regression

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df2['bmi'],df2['age'],df2['charges'],c='red', marker='o', alpha=0.5)
ax.plot_surface(x_surf,y_surf,fittedY.reshape(x_surf.shape), color='b', alpha=0.3)
ax.set_xlabel('bmi')
ax.set_ylabel('age')
ax.set_zlabel('charges')
plt.show()

 
 



