# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 18:30:06 2022

@author: lezy daniel
"""
1.)LAB
import numpy as np
#1
x = np.array([1,2,3,4,5,6,7])
#2
print(x[1])
print(x[3])
print(x[6])
#3
print(np.shape(x))
#4
x = np.array([[1,2,3,4,5,6,7]]).T
print(x)
#5
M = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]])
print(M)
#6
print(np.shape(M))
#7
M = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]]).T
print(M)
#9
print(M[0])
#8
print(M[:,0])
#10
print(np.array([[0,0,0,0,0,0,0]]*7))
print(np.array([[1,1,1]]*3))

2.)LAB
import numpy as np 
#1.Define and print a 6 dimensional vector
vc = np.arange(1,37).reshape(6,6)
print(vc)

#2. Print the transpose of the above vector
print(vc.T)

#3.Define two non square matrices such that they can be mulplied.
M = np.array([[1,2,3],[4,5,6]])
print(M)
N = np.array([[9,8],[7,6],[5,4]])
print(N)

#4.Print the shape of the above matrices
print(np.shape(M))
print(np.shape(N))

#5. Print the product of above two matrices (do so without using the inbuilt functions).
result = [[0,0],[0,0]]

for i in range(len(M)):
    for j in range(len(N[0])):
        for k in range(len(N)):
           result[i][j] = result[i][j]+(M[i][k]*N[k][j])
        
for r in result:       
    print(r)
    
#6.Define two non square matrices of same order and print their sum.    
A = np.array([[1,2],[3,4],[5,6]])
print(A)
B = np.array([[7,8],[9,10],[11,12]])
print(B)
print(A+B)

#7.Define a square matrix A
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(A)

#8.Print the transpose of A
print(A.T)

#9.Print the identity matrix of the above order I.
I = np.identity(3)
print(I)

#10.Verify A.I = I.A for matrix multiplication
result = [[0,0,0],[0,0,0],[0,0,0]]
result= np.dot(A,I)
for r in result:
    print('A.I =',r)
   
    
result = np.dot(I,A)
for r in result:
    print('I.A =',r)
print(" Therefore, A.I = I.A")

#11.Define another square matrix of the same order as A
B = np.array([[11,12,13],[14,15,16],[17,18,19]])
print(B)

#12.Print the product of the matrices as matrix multiplication
print(A@B)
  
#13.Print the product of the matrices by element wise multiplication
print(np.multiply(A,B))

#14.Calculate and print the inverse of A. (Use linalg)
invrsc = np.linalg.det(A)
print(invrsc)

if invrsc == 0:
    print("Inverse does not exist")
else:
    print("Inverse exist")
print(np.linalg.inv(A))


3.)LAB
import pandas as pd
import numpy as py
import seaborn as sns
import matplotlib.pyplot as plt
inc = pd.read_csv('C:/Users/lezy daniel/OneDrive/Desktop/daniel2/insurance.csv')
print(inc.head())
print(inc.describe())
sns.countplot(x = 'smoker', data= inc)
plt.show()
sns.scatterplot(x= 'sex',y= 'age',hue='smoker', data=inc)
plt.show()
sns.pairplot(inc.drop(['region'], axis = 1), hue='smoker', height=2)
plt.show()
sns.boxenplot('charges', data = inc)
plt.show()
sns.heatmap(inc.corr(), data = inc)
plt.show()
x = inc.corr(method = 'pearson')
print(x)
print(inc.head())


4.)LAB
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

 
os.chdir("C:/Users/lezy daniel/OneDrive/Desktop/daniel2")
car = pd.read_csv('CarPrice_Assignment-1.csv')
car=car.drop(columns=['CarName','symboling','carheight','stroke','compressionratio','peakrpm','citympg','highwaympg'])
print(car.head())
print(car.describe())
sns.countplot(x='enginetype', data=car)
plt.show()
sns.scatterplot(x='enginetype',y ='price',data = car)
plt.show()
sns.scatterplot(x='carlength',y='price',data=car)
plt.show()
sns.scatterplot(x='carwidth',y='price',data=car)
plt.show()
sns.pairplot(car.drop(['car_ID'],axis=1),height=3)
plt.show()
x=car.corr(method='pearson')
print(x)
sns.heatmap(car.corr(method='pearson').drop(['car_ID'],axis=1).drop(['car_ID'],axis=0))
plt.show()
sns.boxplot(y="price",data=car)
plt.show()
sns.boxplot(x='enginetype',y='price',data=car)
plt.show()


#one variable regression
x = car[['price']]
y = car[['horsepower']]
reg = linear_model.LinearRegression()
reg.fit(x,y)
print(reg.coef_)
sns.regplot(x,y)


#multiple regression
x = car[['horsepower','enginesize']]
y = car[['price']]
reg = linear_model.LinearRegression()
reg.fit(x,y)
print(reg.coef_)


5.)LAB
getwd()
setwd("C:/Users/lezy daniel/OneDrive/Desktop/daniel2")
car1=read.csv("CarPrice_Assignment-1.csv")
View(car1)
mean(car1$carwidth)
#mean = 65.9078
#Ho: mean of carwidth is equal to 65 
#H1: mean of carwidth is not equal to 65
t.test(car1$carwidth,mu=65,alternative = "less",conf.level = 0.95)


6.)LAB
import pandas as pd
import os
from sklearn.datasets import load_iris
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt

os.chdir("C:/Users/lezy daniel/OneDrive/Desktop/daniel2")
df= pd.read_csv("bfi.csv")
df.columns
# Dropping unnecessary columns
df.drop(['gender', 'education', 'age'],axis=1,inplace=True)
# Dropping missing values rows
df.dropna(inplace=True)

df.info()
df.head()

from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value,p_value=calculate_bartlett_sphericity(df)
chi_square_value, p_value
from factor_analyser.factor_analyser import calculate_kmo
kmo_all,kmo_model=calculate_kmo(df)
kmo_model

# Create factor analysis object and perform factor analysis
fa = FactorAnalyzer()
fa.analyze(df, 25, rotation=None)
# Check Eigenvalues
ev, v = fa.get_eigenvalues()
ev

# Create scree plot using matplotlib
plt.scatter(range(1,df.shape[1]+1),ev)
plt.plot(range(1,df.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()

# Create factor analysis object and perform factor analysis
fa = FactorAnalyzer()
fa.analyze(df, 6, rotation="varimax")
fa.loadings

# Create factor analysis object and perform factor analysis using 5 factors
fa = FactorAnalyzer()
fa.analyze(df, 5, rotation="varimax")
fa.loadings

# Get variance of each factors
fa.get_factor_variance()

 
7.)LAB
#Logistic regression using titanic dataset
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#load dataset
os.chdir("C:/Users/lezy daniel/OneDrive/Desktop/daniel2")
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

# split X and y into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

# instantiate the model (using the default parameters)
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


8.)LAB
#Clustering

import os
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering 
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram,linkage

 
os.chdir("C:/Users/lezy daniel/OneDrive/Desktop/daniel2")
df = pd.read_csv("Shopping_CustomerData.csv")
df.head()
df.info()
genre = pd.get_dummies(df['Genre'])
df.drop(['Genre'],axis=1,inplace=True)
df = pd.concat([df,genre],axis=1)
df.info()
df=np.array(df)

Z = linkage(df, method = "ward") 
dendro = dendrogram(Z) 
plt.title('Dendogram') 
plt.ylabel('Euclidean distance') 
plt. show()

ac=AgglomerativeClustering(n_clusters=2,affinity="euclidean",linkage="ward") 
labels=ac.fit_predict(df) 
plt.figure(figsize = (8,5)) 
plt.scatter(df[labels == 0,0], df[labels == 0,1], c='red')
plt.scatter(df[labels == 1,0], df[labels == 1,1], c='blue') 
plt.scatter(df[labels == 2,0], df[labels == 2,1], c='green') 
plt.scatter(df[labels == 3,0], df[labels == 3,1], c='black') 
plt.scatter(df[labels == 4,0], df[labels == 4,1], c='orange') 
plt.show()