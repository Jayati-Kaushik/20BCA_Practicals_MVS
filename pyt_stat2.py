# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 10:42:55 2021

@author: Raisa
"""
import numpy as np 
#1)Define and print a 6 dimentional vector
a=np.array([[[[[[1,2,3]]]]]])
print(a)
print(a.ndim)
#2)Print the transpose of the above vector
print(a.T)
#3)Define two non square matrices such that they can be mulplied.
x=np.array([[3,4]])
y=np.array([[2],[4]])
print(x)
print(y)
#4) Print the shape of the above matrices
print(x.shape)
print(y.shape)
#5)Print the product of above two matrices (do so without using the inbuiltfunctions).
z=np.array([[0]])
for i in range(len(x)):
    for j in range(len(y[0])):
        for k in range(len(y)):
            z[i][j]+=x[i][k]*y[k][j]
for r in z:
    print(r)
#6)Define two non square matrices of same order and print their sum.
p=np.array([[2,4,3,4],[1,2,3,4]])
q=np.array([[4,5,6,7],[4,3,2,1]])
print(p)
print(q)
print(p+q) 
#7)Define a square matrix A.
A=np.array([[1,2,3],[7,3,9],[3,8,10]])
print(A) 
#8) Print the transpose of A.
print(A.T)
#9)Print the identity matrix of the above order I.
I=np.identity(3,int)
print(I)
#10) Verify A.I = I.A for matrix multiplication.
print(np.dot(A,I))#A.I
print(np.dot(I,A))#I.A
#therefore A.I = I.A
#11) Define another square matrix of the same order as A
B=np.array([[5,6,0],[1,4,1],[0,2,1]])
print(B)
#12)Print the product of the matrices as matrix multiplication
print(np.dot(A,B))
#13) Print the product of the matrices by element wise multiplication
print(4*B)
print(A*B)
#14)  Calculate and print the inverse of A. (Use linalg)
#Check if determinant is 0
det=print(np.linalg.det(A))
#Use if else statement to calculate inverse only when determinant isnon 0
if det!=0:
    print('inverse exists')
    print(np.linalg.inv(A))
else:
    print('inverse does not exixts')



        
    