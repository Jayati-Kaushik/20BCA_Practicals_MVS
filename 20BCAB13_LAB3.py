import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
ins = pd.read_csv('insurance.csv')
print(ins.head())
print(ins.describe())
sns.countplot(x = 'smoker', data= ins)
plt.show()
sns.scatterplot(x= 'sex',y= 'age',hue='smoker', data=ins)
plt.show()
sns.pairplot(ins.drop(['region'], axis = 1), hue='smoker', height=2)
plt.show()
sns.boxenplot('charges', data = ins)
plt.show()
sns.heatmap(ins.corr(), data = ins)
plt.show()
x = ins.corr(method = 'pearson')
print(x) 
sns.heatmap(ins.corr(method='pearson'))
plt.show()