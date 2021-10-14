# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 15:30:22 2021

@author: josha
"""
import numpy as np
# 1. Define and print a 6 dimentional vector
a=np.array([[1,3,2,6,5,7]])
print(a)
# 2. Print the transpose of the above vector
print(a.T)
# 3. Define two non square matrices such that they can be mulplied.
x=np.array([[1,3],[2,6],[5,7]])
y=np.array([[1,3,2],[6,5,7]])
# 4. Print the shape of the above matrices
print(np.shape(x))
print(np.shape(y))
# 5. Print the product of above two matrices(without inbuilt fn)
O=np.array([np.zeros(3)]*3)
for i in range(len(x)):
    for j in range(len(y[1])):
        for k in range(len(y)):
            O[i][j] += x[i][k] * y[k][j]       
print(O)
# 6. Define two square matrices of same order and print their sum.
m=np.array([[1,3],[2,6]])
n=np.array([[2,5],[4,3]])
print(m+n)
# 7. Define a square matrix A.
A=np.array([[1,2,3],[4,5,6],[7,8,9]])
# 8. Print the transpose of A.
print(A.T)
# 9. Print the identity matrix of the above order I.
I=np.array([[1,0,0],[0,1,0],[0,0,1]])
print(I)
# 10. Verify A.I = I.A for matrix multiplication.
p=np.dot(A,I)
q=np.dot(I,A)
print(p)
print(q)
print("hence both are equal")
# 11. Define another square matrix of the same order as A.
B=np.array([[9,8,7],[6,5,4],[3,2,1]])
# 12. Print the product of the matrices as matrix multiplication+
j=np.dot(A,B)
print(j)
# 13. Print the product of the matrices by element wise multiplication
print(np.multiply(A,B))
# 14. Calculate and print the inverse of A. (Use linalg)
DET=np.linalg.det(A)
print(DET)
if DET!=0:
    print("inverse exist")
    print(np.linalg.inv(A))
else:
    print("inverse doesnt exist")







