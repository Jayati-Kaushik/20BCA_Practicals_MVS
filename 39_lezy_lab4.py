import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

 
os.chdir("C:/Users/lezy daniel/OneDrive/Desktop/daniel2")
car = pd.read_csv('CarPrice_Assignment-1.csv')
car=car.drop(columns=['CarName','symboling','carheight','stroke','compressionratio','peakrpm','citympg','highwaympg'])
print(car.head())
print(car.describe())
sns.countplot(x='enginetype', data=car)
plt.show()
sns.scatterplot(x='enginetype',y ='price',data = car)
plt.show()
sns.scatterplot(x='carlength',y='price',data=car)
plt.show()
sns.scatterplot(x='carwidth',y='price',data=car)
plt.show()
sns.pairplot(car.drop(['car_ID'],axis=1),height=3)
plt.show()
x=car.corr(method='pearson')
print(x)
sns.heatmap(car.corr(method='pearson').drop(['car_ID'],axis=1).drop(['car_ID'],axis=0))
plt.show()
sns.boxplot(y="price",data=car)
plt.show()
sns.boxplot(x='enginetype',y='price',data=car)
plt.show()

#one variable regression
x = car[['price']]
y = car[['horsepower']]
reg = linear_model.LinearRegression()
reg.fit(x,y)
print(reg.coef_)
sns.regplot(x,y)


#multiple regression
x = car[['horsepower','enginesize']]
y = car[['price']]
reg = linear_model.LinearRegression()
reg.fit(x,y)
print(reg.coef_)
