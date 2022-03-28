# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 22:41:53 2022

@author: david
"""

import numpy as np
# 1. Define and print a 6 dimentional vector
X=np.array([[1,2,3,4,5,6]])
print(X)

# 2. Print the transpose of the above vector
print(X.T)

# 3. Define two non square matrices such that they can be mulplied.
X=np.array([[1,2],[3,4],[5,6]])
Y=np.array([[1,2,3],[4,5,6]])

# 4. Print the shape of the above matrices\
print(np.shape(X), np.shape(Y))

# 5. Print the product of above two matrices (do so without using the inbuilt functions).
Z=np.array([np.zeros(3)]*3)
for i in range(len(X)):
    for j in range(len(Y[1])):
        for k in range(len(Y)):
            Z[i][j] += X[i][k] * Y[k][j]
print(Z)

# 6. Define two non square matrices of same order and print their sum.
A=np.array([[1,2,3],[4,5,6]])
B=np.array([[-1,-2,-3],[-4,-5,-6]])
print(A+B)

# 7. Define a square matrix A.
A=np.array([[7,2,4],[4,9,6],[7,8,9]])

# 8. Print the transpose of A.
print(A.T)

# 9. Print the identity matrix of the above order I.
I=np.array([[1,0,0],[0,1,0],[0,0,1]])
print(I)

# 10. Verify A.I = I.A for matrix multiplication.
X=A*I
print("A.I = ",X)
Y=I*A
print("I.A = ",Y)
print(" Therefore, A.I = I.A")
 
# 11. Define another square matrix of the same order as A.
B=np.array([[2,5,7],[3,6,3],[0,1,9]])

# 12. Print the product of the matrices as matrix multiplication
print(A*B)

# 13. Print the product of the matrices by element wise multiplication
print(np.multiply(A,B)) 

# 14. Calculate and print the inverse of A. (Use linalg)
d=np.linalg.det(A)
print("Determinant = ",d)  
if d!=0:
    print("Inverse of A = ",np.linalg.inv(A))
else:
    print("Inverse does not exist")    