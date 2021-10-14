# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 12:14:24 2021

@author: joelp
"""

import numpy as np

#1 Define and print a 6 dimentional vector
x = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
type(x)

#2 Print the transpose of the above vector
print(x.T)

#3 Define teo non square matrices such that they can be multiplied
a=np.array([[2,5,4],[1,7,4]])
b=np.array([[6,7],[2.5],[1,2]])
print(a)
print(b)

#4 Print te shape of the above matrices
print(a.shape)
print(b.shape)

#5 Print the product of above two matrices
Z=np.array([np.zeros(3)*3])
for i in range(len(a)):
    for j in range(len(b[1])):
        for k in range(len(b)):
            Z[i][j] = a[i][j] * b[i][j]
print(Z)           

#6 Define two non square matrices and print their sum
p=np.array([[4,6,8],[2,6,3],[9,8,7]])
q=np.array([[4,1,7],[7,8,9],[5,8,6]])
print(p+q)

#7 Define a square matrix A
A=np.array([[5,3],[9,3]])
print(A)

#8 Print the transpose of A 
print(A.T)

#9 Print the identity matrix of the above order I
I=np.array([[1,0],[0,1]])
print(I)

#10 Verify A.I = I.A for matrix multiplication
X=A@I
print("A.I = ",X)
Y=A@I
print("I.A = ",Y)
print(" Therefore, A.I = I.A")

#11 Define another square matrix of the same order as A
O=np.array([[7,4],[1,5]])
print(O)
#12 Print the product of the matrices as matrix multiplication
print([7,4]*[1,5])
#13 Print the product of the matrices by element wise multiplication

#14 Calculate and print the inverse of A.