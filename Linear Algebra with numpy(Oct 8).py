# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 12:23:11 2021

@author: chris
"""

import numpy as np

z=np.array([[1,2,3,4,5,6]])
print(z)

#Print the transpose of the above vector
print(z.T)

#Non-square matrices to be multiplied
X=np.array([[1,4],[5,4],[3,6]])
Y=np.array([[1,2,3],[4,5,6]])
print(X)
print(Y)

#Print the shape of the above matrices
print(X.shape)
print(Y.shape)

#Print the product of above two matrices
A=np.array([np.zeros(3)]*3)
for i in range(len(X)):
    for j in range(len(Y[1])):
        for k in range(len(Y)):
            A[i][j] += X[i][k] * Y[k][j]
print(A)

# Define two non square matrices of same order and print their sum
K=np.array([[3,6,1],[9,8,1],[1,5,2]])
T=np.array([[1,8,9],[2,5,6],[5,1,7]])
print(K+T)

#Define a square matrix A.
A=np.array([[2,3],[5,6]])

#Print the transpose of A
print(A.T)

#Print the identity matrix of the above order I.
I=np.array([[1,0],[0,1]])
print(I)

#Verify A.I = I.A for matrix multiplication
X=A@I
print("A.I = ",X)
Y=I@A
print("I.A = ",Y)
print(" Therefore, A.I = I.A")

#Define another square matrix of the same order as A
B=np.array([[1,5],[4,9]])

#Print the product of the matrices as matrix multiplication
print(A@B)

#Print the product of the matrices by element wise multiplication
print(np.multiply(A,B))

#Calculate and print the inverse of A
E=np.linalg.det(A)
print("The Determinant:",E)
if E!=0:
    print("The Inverse is:",np.linalg.inv(A))
else:
    print("The Inverse cannot be found")
    
    



