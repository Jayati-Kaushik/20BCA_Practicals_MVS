# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 15:10:34 2022

@author: Surya
"""
#Logistic Regression
import pandas as pd
#For creating a directory
import os
#Returning the current working directory
os.getcwd()
#Changing the current working directory
os.chdir('C:\\Users\\Surya\\Desktop')
titanic=pd.read_csv('titanic.csv')
#Returns the first five rows
print(titanic.head())
#Converting the data into dataframe
df = pd.DataFrame(titanic)
#To drop the columns
df.drop(['PassengerId','Name','Ticket','Cabin'], axis = 1)
#Selecting the columns
X=titanic.iloc[:, [0,2,5]]
y=titanic.Survived
#using this package for splitting the data into training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
#importing the class
from sklearn.linear_model import LogisticRegression
#instantiating(defining) the model 
logreg = LogisticRegression()
#Replacing missing values in age column with mean
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].mean())
# fitting the model with the data
logreg.fit(X_train,y_train)
#predicting the dependent variable
y_pred=logreg.predict(X_test)
#model evaluation using confusion matrix
#importing the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
#Visualizing confusion matrix using heatmap
sns.heatmap(pd.DataFrame(cnf_matrix),annot=True,cmap='Reds',fmt='g')