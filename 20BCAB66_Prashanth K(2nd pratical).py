# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 17:08:26 2021

@author: prash
"""

import numpy as np
#1) define and print a 6 dimensional vector
a=np.array([[1,2,3,4,5,6],
            [7,8,9,4,5,2],
            [9,3,4,1,5,0],
            [8,1,4,7,8,9],
            [9,7,3,8,4,1],
            [9,1,0,8,3,2]
            ])
print(a)
#2)print the transpose of the above vector
print(a.T)
#3)Define two non square matrices such that they can be multiplied
x=np.array([[1,2,3],
            [4,5,6]])
y=np.array([[2,3],
            [4,5],
            [6,7]])
print(x)
print(y)
#4) Print the shape of the above matrices
print(x.shape)
print(y.shape)
#5)print the product of the above two matrices (without using in built function)
z=np.array([np.zeros(3)]*3)
for i in range(len(x)):
    for j in range(len(y[1])):
        for k in range(len(y)):
            z[i][j]+=x[i][j] * y[k][j]
            
print(z)            
            
#6)define two non square matrices of same order and print their sum
p=np.array([[1,2,3],
            [4,5,6],
            [7,8,9]])
q=np.array([[9,8,7],
            [6,5,4],
            [3,2,1]])
C=p+q
print(C)
#7)Define a square matrix 
A=np.array([[2,2],
           [3,2]
           ])
print(A)
#8)print the transpose of A
print(A.T)
#9)print the identity matrix of the above order I
I=np.array([[1,0],
            [0,1]
            ])
print(I)
#10)Verify A.I = I.A for matrix multiplication
r=A@I
print("A.I=",r)
t=I@A
print("I.A=",t)
print("Therefore,A.I=I.A")
#11)Define another square matrix square order as A
m=np.array([[5,5],
            [2,8]]
           )
print(m)
#12)print the product of the matrices as matrix multiplication
print(A@m)
#13)print the product of the matrices by element wise multiplication
print(np.multiply(A,m))
#14)calculate and print the inverse of A
c=np.linalg.det(A)
print("The determinant of A is ",c)
if c!=0:
    print("The Inverse is",np.linalg.inv(A))
    
else:
    print("Inserve does not exist")

