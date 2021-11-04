import pandas as pd
import seaborn as sbn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
re = pd.read_csv("C:/Users/WinDows/OneDrive/Desktop/realestate.csv")
#print(re.head())
#print(re.isnull().sum())
#print(re.info())
des = re.describe()
#sbn.pairplot(re)
corr=re.corr()
#sbn.heatmap(corr)
#print(re.columns)
x = re[['No', 'X1 transaction date', 'X2 house age',
       'X3 distance to the nearest MRT station',
       'X4 number of convenience stores', 'X5 latitude', 'X6 longitude',]]
y = re['Y house price of unit area']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.30)
lm = LinearRegression()
lm.fit(x_train,y_train)
prediction = lm.predict(x_test)
#plt.scatter(y_test,prediction)
sbn.distplot((y_test-prediction),bins=50)
