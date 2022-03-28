# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 22:47:56 2022

@author: Aarushi
"""

#Program 1:
import numpy as np
#1.Define and print a 6 dimensional vector
A=np.array([[1,2,3,4,5,6],
            [9,3,6,1,0,7],
            [10,5,3,8,6,1],
            [1,7,8,3,2,9],
            [6,0,5,1,7,2],
            [8,4,1,7,3,5]])
print("6 dimensional vector is \n",A)

#2. Print the transpose of the above vector
print("Transpose of the above vector is \n",np.transpose(A))

#3. Define two non-square matrices such that they can be multiplied
A=np.array([[1,2,3],[4,5,6]])
B=np.array([[2,4],[1,6],[1,5]])
print("Two non-square matrices are:")
print(A)
print(B)

#4. Print the shape of the above matrices
print(np.shape(A))

#5.Print the product of above two matrices(without inbuilt function)
print("Product of above matrices")
result=[[0,0],
        [0,0]]
for i in range(len(A)):
    for j in range(len(B[0])):
        for k in range(len(B)):
            result[i][j] = result[i][j]+A[i][k]*B[k][j]
for r in result:
    print (r)

#6. Define two non square matrices of same order and print their sum.
print("Two non square matrices of same order")
X=np.array([[3,2,1],[2,6,8]])
Y=np.array([[4,5,6],[8,12,9]])
print(X)
print(Y)
print("Sum of above matrices")
result=[[0,0,0],
        [0,0,0]]
for i in range(len(X)):
    for j in range(len(Y[0])):
        result[i][j]=X[i][j]+Y[i][j]
for r in result:
    print (r)

#7. Define a square matrix A.
A=np.array([[10,11,12],[5,9,2],[15,8,4]])
print("Square matrix A")
print(A)

#8. Print the transpose of A.
print("Transpose of above matrix\n",np.transpose(A))

#9. Print the identity matrix of the above order I.
print("Identity matrix of same order")
I=np.identity(3)
print(I)

#10. Verify A.I = I.A for matrix multiplication.
A=np.array([[10,11,12],[5,9,2],[15,8,4]])
I=np.identity(3)
print("A*I")
print(np.dot(A,I))
print("I*A")
print(np.dot(I,A))
if(np.dot(A,I).all()==np.dot(I,A).all()):
    print("Verified")
else:
    print("Not verified")

#11. Define another square matrix of the same order as A.
print("Square matrix B of same order as A")
B=np.array([[10,5,7],[6,12,3],[4,6,2]])
print(B)

#12. Print the product of the matrices as matrix multiplication
print("Product of A and B")
print(np.dot(A,B))

#13. Print the product of the matrices by element wise multiplication
print("Product of above matrices by element wise multiplication")
print(np.multiply(A,B))

#14. Calculate and print the inverse of A. (Use linalg)

#a When determinant equal to zero
A=([[1,2,3],[3,5,7],[3,4,5]])
D=np.linalg.det(A)
if (D==0):
    print("Inverse doesnt exist")
else:
    print("Inverse exist")
    print("Inverse of A is")
    print(np.linalg.inv(A))

#b When determinant not equal to zero
A=([[1,2,3],[0,1,4],[5,6,0]])
D=np.linalg.det(A)
if (D==0):
    print("Inverse doesnt exist")
else:
    print("Inverse exist")
    print("Inverse of A is")
    print(np.linalg.inv(A))    
#Program 2: Data visualization
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
iris = pd.read_csv('Iris.csv')
iris.head()
print(iris.head())
print(iris.describe())
sns.countplot(x='class', data = iris)
plt.show()
sns.scatterplot('sepallength','sepalwidth', hue='class',data = iris)
plt.show()
sns.pairplot(iris.drop(['id'],axis =1), hue= 'class', height= 2)
plt.show()
sns.boxenplot()
plt.show()
sns.heatmap(iris.corr(), data = iris)
plt.show()
x= iris.corr(method= 'pearson')
print(x)
#Program 3: Simple linear regression
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data1=pd.read_csv('CarPrice_Assignment.csv')
data1=data1.drop(columns=['CarName','symboling','carheight','stroke','compressionratio','peakrpm','citympg','highwaympg'])
print(data1.head())
print(data1.describe())
sns.countplot(x='enginetype', data=data1)
plt.show()
sns.scatterplot(x='enginetype',y ='price',data =data1)
plt.show()
sns.scatterplot(x='carlength',y='price',data=data1)
plt.show()
sns.scatterplot(x='carwidth',y='price',data=data1)
plt.show()
sns.pairplot(data1.drop(['car_ID'],axis=1),height=3)
plt.show()
x=data1.corr(method='pearson')
print(x)
sns.heatmap(data1.corr(method='pearson').drop(['car_ID'],axis=1).drop(['car_ID'],axis=0))
plt.show()
sns.boxplot(y="price",data=data1)
plt.show()
sns.boxplot(x='enginetype',y='price',data=data1)
plt.show()

#One variable regression
X= data1[['enginesize']]
Y= data1[['price']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)
reg = LinearRegression()
reg.fit(X_train,Y_train)
Y_pred = reg.predict(X_test)
# The coefficients
print("Coefficients: \n", reg.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(Y_test, Y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(Y_test, Y_pred))
# Variance score
print("Variance score: {}".format(reg.score(X_test,Y_test)))
#sns.scatterplot(X,Y,data=data1)
sns.regplot(X,Y)

#Program 4: Multiple regression
X= data1[['wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower','carlength','carwidth']]
Y= data1['price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)
reg = LinearRegression()
reg.fit(X_train,Y_train)
Y_pred = reg.predict(X_test)
# The coefficients
print("Coefficients: \n", reg.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(Y_test, Y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(Y_test, Y_pred))
# Variance score
print("Variance score: {}".format(reg.score(X_test,Y_test)))

#Program 5: Hypothesis Testing
#hypothesis testing
setwd("C: /Users/arcot/Desktop/R Studio ")
data1=read.csv("CarPrice_Assignment.csv")
View(data1)
mean(data1$carlength)
#Ho: mean of carlength is equal to 170
#H1: mean of carlength is not equal to 170
t.test(data1$carlength,mu=170,alternative="less",conf.level=0.95)

#Program 6: Factor Analysis
import os
import pandas as pd
import scipy as sp
import numpy as np
import sklearn as kl
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
import factor_analyzer
os.chdir("C:/Users/arcot/Desktop/R Studio")
df= pd.read_csv("bfi.csv")
print(df)
df.head()
 
df.columns 
df.drop(['gender','education','age'],axis=1,inplace=True)
df.dropna(inplace=True)
df.info()
df.head()
factor_analyzer.factor_analyzer.calculate_kmo(df)
fa = FactorAnalyzer(n_factors=25,rotation=None)
fa.fit(df)
ev,cev=fa.get_eigenvalues()
plt.scatter(range(1,df.shape[1]+1),ev)
plt.plot(range(1,df.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()
fa = FactorAnalyzer(n_factors=7,rotation="varimax")
fa.fit(df)
fa.loadings_
fa.get_communalities()
fa.get_eigenvalues()
fa.get_factor_variance()
#Program 7: Logistic Regression
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#load dataset
os.chdir("C:/Users/arcot/Desktop/R Studio ")
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

#Program 8: Hierarchical clustering
from sklearn.cluster import AgglomerativeClustering
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage 

os.chdir("C:/Users/arcot/Desktop/R Studio")
data = pd.read_csv("CarPrice_Assignment.csv")
data.drop(['car_ID','CarName'],axis=1,inplace=True)
data.info()
df = data.iloc[:, [8,9]].values
Z = linkage(df, method = "ward")
dendro = dendrogram(Z)
plt.title('Dendogram')
plt.ylabel('Euclidean distance')
plt.show()
ac = AgglomerativeClustering(n_clusters=4, affinity="euclidean", linkage="ward")

labels = ac.fit_predict(df)
plt.figure(figsize = (8,5))
plt.scatter(df[labels == 0,0] , df[labels == 0,1], c= 'red')
plt.scatter(df[labels == 1,0] , df[labels == 1,1], c= 'blue')
plt.scatter(df[labels == 2,0] , df[labels == 2,1], c= 'green')
plt.scatter(df[labels == 3,0] , df[labels == 3,1], c= 'black')
plt.scatter(df[labels == 4,0] , df[labels == 4,1], c= 'orange')
plt.show()
