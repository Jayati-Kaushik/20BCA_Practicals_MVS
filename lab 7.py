import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


os.chdir("C:/Users/prach/OneDrive/Desktop/admin")
dataset = pd.read_csv('User_Data.csv')
#dataset.drop(['User_ID','Gender'])

#to predict whether a user will purchase the product or not,
#we have to find out the relationship between Age and Estimated Salary. 

# input
x = dataset.iloc[:, [2, 3]].values

# output
y = dataset.iloc[:, 4].values

#split data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25, random_state = 0)

#feature scaling age and salary as they lie in different ranges
#if not done, salary will dominate age when model finds the nearest neighbor to a data point in data space.
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
xtest = sc_x.transform(xtest)
print (xtrain[0:10, :])
#o/p ranges from -1 to 1, equal contribution in finalizing hypothesis

#train our model
classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain, ytrain)

#use model on test data
y_pred = classifier.predict(xtest)

#test performance of model on confusion matrix
cm = confusion_matrix(ytest, y_pred)
print ("Confusion Matrix : \n", cm)
# [[65  3]   TP FP
#  [ 8 24]]  FN TN

# accuracy
print ("Accuracy : ", accuracy_score(ytest, y_pred))
# 0.89
# which is pretty good