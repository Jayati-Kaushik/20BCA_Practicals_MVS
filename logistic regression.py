# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 17:36:04 2022

@author: Harshitha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv('User_Data.csv')

# input
x = dataset.iloc[:, [2, 3]].values

# output
y = dataset.iloc[:, 4].values
#splitting the data set
xtrain, xtest, ytrain, ytest = train_test_split(
		x, y, test_size = 0.25, random_state = 0)

sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
xtest = sc_x.transform(xtest)
print (xtrain[0:10, :])
#training the logistic regression model
classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain, ytrain)
y_pred = classifier.predict(xtest)
#confusion matrix
cm = confusion_matrix(ytest, y_pred)
print ("Confusion Matrix : \n", cm)
from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(ytest, y_pred))



