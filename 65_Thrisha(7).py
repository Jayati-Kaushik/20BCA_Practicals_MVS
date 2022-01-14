# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 20:27:08 2022

@author: DEC BEAST
"""
#importing packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

#loading the dataset
data1 = pd.read_csv('titanic.csv')
#dropping the rows with na values
data1 = data1.dropna()

data1.info()
data1.head()

#split dataset in features and target variable
feature_cols = ['PassengerId','Age','Parch','Fare','Pclass']
X= data1[feature_cols] #features
Y= data1.Survived #Target variable

#splitting x and y as training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)

#instantiate the model (using default parameters)
logreg = LogisticRegression()

#fit the model with data
logreg.fit(X_train,Y_train)

Y_pred = logreg.predict(X_test)

#confusion matrix
cnf_matrix = metrics.confusion_matrix(Y_test, Y_pred)
cnf_matrix

#Visualizing confusion matrix
#create heatmap 
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True ,fmt='g')
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual')
plt.xlabel('Predicted')

#evaluation metrics
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("Precision:",metrics.precision_score(Y_test, Y_pred))
print("Recall:",metrics.recall_score(Y_test, Y_pred))