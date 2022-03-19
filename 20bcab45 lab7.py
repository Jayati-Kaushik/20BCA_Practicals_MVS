# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 21:09:12 2022

@author: ADHIR
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#load dataset
os.chdir("C:/Users/Adhir/Desktop/DATASETS")
data = pd.read_csv("titanic.csv")
data.head()
data.info()

#creating dummies for categorical variables
sex = pd.get_dummies(data['Sex'])
embark = pd.get_dummies(data['Embarked'])
data.drop(['Sex','Embarked','Name','Ticket','Cabin','Age'],axis=1,inplace=True)
data = pd.concat([data,sex,embark],axis=1)
data.info()

#split dataset into X and Y
X = data.drop(['Survived'], axis=1)
y = data['Survived']

# spliting X and y into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

logreg = LogisticRegression(solver='lbfgs', max_iter=1000)

# fit the model with data
logreg.fit(X_train,y_train)

#prediction
y_pred=logreg.predict(X_test)

#confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
print(cnf_matrix)

#eavluation metrics
print(classification_report(y_test,y_pred))
