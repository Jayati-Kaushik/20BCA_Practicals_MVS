# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 09:45:25 2022

@author: Joel Thomas
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

os.chdir("C:/Users/Rose Maria Thomas/Desktop/joel")
data=pd.read_csv('titanic.csv')
data.info()


#analysis
sns.countplot(x="Survived",data=data)
sns.countplot(x="Survived",hue="Sex",data=data)
sns.countplot(x='Survived',hue="Pclass",data=data)
data['Age'].plot.hist()
sns.countplot(x="SibSp",data=data)
sns.countplot(x="Parch",data=data)

#Cleaning dataset
data.isnull().sum()
sns.boxplot(x="Pclass",y="Age",data=data)
data.drop("Cabin",axis=1,inplace=True)
data.drop("Age",axis=1,inplace=True)
data.drop("Embarked",axis=1,inplace=True)

#convert string values to float 
sex=pd.get_dummies(data["Sex"],drop_first=True)
pcl=pd.get_dummies(data["Pclass"],drop_first=True)
dat=pd.concat([data,sex,pcl],axis=1)
dat.drop(["Sex","PassengerId","Name","Ticket","Pclass"],axis=1,inplace=True)

#Test and train data
x=dat.drop("Survived",axis=1)
y=dat["Survived"]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=1)

#Logistic regression
logreg=LogisticRegression()
logreg.fit(xtrain,ytrain)
pred=logreg.predict(xtest)

#Confusion matrix
confusion_matrix(ytest,pred)
#output-array
#([[133,  20],[ 45,  70]], dtype=int64)

#Accuracy
accuracy_score(ytest,pred)
#Output-0.757