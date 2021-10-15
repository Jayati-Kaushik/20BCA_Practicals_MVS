# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 10:14:33 2021

@author: Harshitha
"""

import numpy as np
#1
A=np.array([[1,3,3,4,5,6],
            [9,6,6,2,0,7],
            [10,5,3,8,6,1],
            [1,7,8,3,2,8],
            [6,1,5,9,7,2],
            [8,4,2,7,3,5]])
print(A)
#2 TRANSPOSE
print(np.transpose(A))
#3 non square matrices which can be multiplied
A=np.array([[3,2,1],[7,5,6]])
B=np.array([[1,4],[1,4],[1,4]])
print("Two non-square matrices are:")
print(A)
print(B)
#4 shape of above matrices
print(np.shape(A))
print(np.shape(B))
#5 product of two matrices without using inbulit function

result=[[0,0],
        [0,0]]
for i in range(len(A)):
    for j in range(len(B[0])):
        for k in range(len(B)):
            result[i][j] = result[i][j]+A[i][k]*B[k][j]
for r in result:
    print (r)
    
#6Defining two non square matrices of same order and print their sum.

X=np.array([[1,2,1],[2,4,8]])
Y=np.array([[4,8,6],[8,12,16]])
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
#7 defining square matrix
A=np.array([[19,11,12],[1,9,2],[5,8,14]])
print(A)
#8 tranpose of A
print(np.transpose(A))
#9 identity matrix of same order
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
    
#11. Defining another square matrix of the same order as A.
print("Square matrix B of same order as A")
B=np.array([[1,5,17],[16,2,3],[4,6,12]])
print(B)

#12. Printing the product of the matrices as matrix multiplication
print("Product of A and B")
print(np.dot(A,B))

#13. Printing the product of the matrices by element wise multiplication
print("Product of above matrices by element wise multiplication")
print(np.multiply(A,B))

#14. Calculate and print the inverse of A. (Use linalg)

#a When determinant equal to zero
A=([[1,12,3],[3,15,7],[6,4,15]])
D=np.linalg.det(A)
if (D==0):
    print("Inverse doesnt exist")
else:
    print("Inverse exist")
    print("Inverse of A is")
    print(np.linalg.inv(A))