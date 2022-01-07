import pandas as pd
import matplotlib.pyplot as plt
#helps in creating a dictionary like object
from sklearn.datasets import load_iris
#For creating a directory
import os
#Returning the current working directory
cwd = os.getcwd()
#Changing the current working directory
os.chdir('C:/Users/SONY/Documents/weather')
weather=pd.read_csv('weather.csv')
print(weather)
#Subsetting the data
x =weather[weather.columns[12:17]]
from factor_analyzer import FactorAnalyzer
fa = FactorAnalyzer()
fa.fit(x, 6)
#Getting Eigen values and plotting them
ev, v = fa.get_eigenvalues()
print(ev)
plt.plot(range(1,x.shape[1]+1),ev)
#Using varimax rotation for performing factor analysis
fa = FactorAnalyzer(4, rotation='varimax')
fa.fit(x)
#A good factor loading value is 0.5
loads = fa.loadings_
print(loads)
#Getting the variance of each factors
fa.get_factor_variance()
