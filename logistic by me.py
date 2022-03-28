import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
d1= load_iris()
d1= pd.DataFrame(data= np.c_[d1['data'], d1['target']], columns = d1['feature_names']+['Species'])
print(d1.head())
print(d1.info())
print(d1.describe())
plt.ylabel('sepal length (cm)')
plt.ylabel('petal length (cm)')
plt.plot(d1['sepal length (cm)'], d1['petal length (cm)'])
plt.show()
plt.bar(d1['sepal length (cm)'],d1['petal length (cm)'] )
plt.show()
x = d1.iloc[:,[0,1,2,3]].values
print(x)
y= d1.iloc[:,4].values
print(y)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.20, random_state = 0)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, solver='lbfgs', multi_class='auto')
classifier.fit(x_train, y_train)