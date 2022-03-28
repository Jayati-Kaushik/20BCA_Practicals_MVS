# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 22:38:19 2022

@author: fiona
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


os.chdir("C:/Users/Lohith/Documents")
data=pd.read_csv('Telecom_Data.csv')

       
data.info()


# regressor variables 
x = data.iloc[:, 0:20].values
#print(x)
  
# regressed variables
y = data.iloc[:, 20].values
#print(y)


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25, random_state = 0)




classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain, ytrain)

#y_pred = classifier.predict(xtest)

#cm = confusion_matrix(ytest, y_pred)
  
#print ("Confusion Matrix : \n", cm)



#Alternative my method
#import pandas as pd
#import numpy as np
#from sklearn import datasets
#from matplotlib import pyplot as plt
#import seaborn as sns
#from sklearn.linear_model import LogisticRegression
#iris=datasets.load_iris()
#iris=pd.DataFrame(data=np.c_[iris['data'],iris['target']],columns=iris['feature_names']+['species'])
#print(iris.head(10))
#print(iris.shape)
#iris.info()
#print(iris.describe())
#plt.bar(iris['sepal length (cm)'],iris['sepal width (cm)'])
#plt.title('Bar Chart')
#plt.xlabel('Sepal Length')
#plt.ylabel('Sepal Width')
#plt.show()
#sns.countplot(x="sepal length (cm)",hue='species',data=iris)
#X, y = load_iris(return_X_y=True)
#clf = LogisticRegression(random_state=0).fit(X, y)
#clf.predict(X[:2, :])