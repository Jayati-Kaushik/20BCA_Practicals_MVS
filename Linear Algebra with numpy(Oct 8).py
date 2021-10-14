# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 10:54:49 2021

@author: FIONA
"""

import numpy as np
#1. Define and print a 6 dimentional vector
x=np.array([[1,1,1,1,1,1],[2,2,2,2,2,2],[3,3,3,3,3,3],[4,4,4,4,4,4],[5,5,5,5,5,5],[6,6,6,6,6,6]])
print(x)

#2. Print the transpose of the above vector
X=np.array([[1,1,1,1,1,1],[2,2,2,2,2,2],[3,3,3,3,3,3],[4,4,4,4,4,4],[5,5,5,5,5,5],[6,6,6,6,6,6]]).T
print(X)

#3. Define two non square matrices such that they can be mulplied.
a=np.array([[1,2,6],[8,2,9]])
b=np.array([[2,5],[8,1],[2,6]])
print(a)
print(b)

#4. Print the shape of the above matrices
print(a.shape)
print(b.shape)

#5. Print the product of above two matrices (do so without using the inbuilt functions).
Z=np.array([np.zeros(3)]*3)
for i in range(len(a)):
    for j in range(len(b[1])):
        for k in range(len(b)):
            Z[i][j] += a[i][k] * b[k][j]
print(Z)

#6. Define two non square matrices of same order and print their sum.
p=np.array([[5,3,9],[9,2,4],[9,4,7]])
q=np.array([[9,3,6],[5,2,8],[8,1,2]])
print(p+q)

#7. Define a square matrix A.
A=np.array([[7,3],[9,2]])
print(A)

#8. Print the transpose of A.
print(A.T)

#9. Print the identity matrix of the above order I.
I=np.array([[1,0],[0,1]])
print(I)

#10. Verify A.I = I.A for matrix multiplication.
X=A@I
print("A.I = ",X)
Y=I@A
print("I.A = ",Y)
print(" Therefore, A.I = I.A") 

#11. Define another square matrix of the same order as A.       
m=np.array([[4,1],[7,2]])
print(m)

#12. Print the product of the matrices as matrix multiplication
print(A@m)

#13. Print the product of the matrices by element wise multiplication
print(np.multiply(A,m))    

#14. Calculate and print the inverse of A.
c=np.linalg.det(A)
print("The determinant of A is",c)
if c!=0:
    print("The inverse is ",np.linalg.inv(A))
else:
    print("Inserve does not exist")