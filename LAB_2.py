# -*- coding: utf-8 -*-
"""
Created on Wed Oct  13 19:22:43 2021

@author: Sohan Immanuel
"""

import numpy as np

#Define and print 6D vector.
X = np.array([[1,0,9,8,7,6],[9,5,7,3,1,1],[8,5,9,4,2,5],[9,6,8,3,9,7],[7,3,2,1,3,5],[6,3,1,1,9,8]])
v1 = np.array(X)
print(v1)

#Print the transpose of the above vector.
print(v1.T)
#Define two non square matrices such that they can be multiplied.
A1 = np.array([[9,1,1],[1,0,1]])
print(A1)
B1 = np.array([[8,9,9],[1,3,2]])
print(B1)

#Print the shape of the above matrix.
print(np.shape(A1))
print(np.shape(B1))

#Print the product of above two matrices
result = [[0,0,0],[0,0,0]]

for i in range(len(A1)):
    for j in range(len(B1[0])):
        for k in range(len(B1)):
           result[i][j] = result[i][j]+(A1[i][k]*B1[k][j])

for r in result:       
    print(r)

#Define two non square matrix and print their sum.
result = [[0,0,0],[0,0,0]]
a1 = np.array([[6,8,9],[1,2,9]])
print(a1)
a2 = np.array([[1,2,3],[4,5,6]])
print(a2)

for i in range(len(a1)):
    for j in range(len(a1[0])):
        result[i][j] = a1[i][j] + a2[i][j]

for r in result:
    print(r)


#Define a square matrix A.
A = np.array([[2,3],[5,6]])
print(A)

#Print the transpose of A
print(A.T)

#Print the identity matrix of the above order I
I = np.identity(2)
print(I)

#Verify A.I=I.A for matrix multiplication.
result = [[0,0],[0,0]]
result= np.dot(A,I)
for r in result:
    print('A.I =',r)

result = np.dot(I,A)
for r in result:
    print('I.A =',r)
print(" Therefore, A.I = I.A")

#Define another square matrix of the same order as A
B = np.array([[3,9],[1,1]])
print(B)

#Print the product of the matrices as matrix multiplication.
print(A@B)

#Print the product of the matrices by element wise multiplication.
print(np.multiply(A,B))

#Calculate and print the inverse of A
D=np.linalg.det(A)
print("The Determinant:",D)
if D!=0:
    print("The Inverse is:",np.linalg.inv(A))
else:
    print("The Inverse cannot be found") 

