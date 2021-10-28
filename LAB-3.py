import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 


iris = pd.read_csv('Iris.csv')

#print(iris.head())
#print(iris.describe())
#sns.countplot(x = 'Species', data= iris)
#sns.scatterplot('SepalLengthCm', 'SepalWidthCm', hue='Species', data=iris)
#sns.pairplot(iris.drop(['Id'], axis = 1), hue='Species', height=2)
#sns.boxenplot()
#sns.heatmap(corr_matrix(), data = iris)
#X = iris.corr(method='pearson')
#print(X)
#sns.heatmap(iris.corr(method='pearson').drop(['Id'],axis=1).drop(['Id'],axis=0))
#plt.show()

