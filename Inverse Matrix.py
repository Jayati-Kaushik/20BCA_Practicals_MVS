# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 23:04:18 2022

@author: SharwinA
"""

import numpy as np

#1. Define and print a 6 dimentional vector
A=np.array([[1,2,3,4,5,6],
            [7,2,9,4,3,7],
            [10,3,2,4,8,3],
            [2,5,4,1,6,9],
            [1,4,6,8,3,2],
            [2,6,8,4,3,6]])

print("6 dimentional vector is\n",A)



#2. Print the transpose of the above vector
print("Transpose of the above vector\n",np.transpose(A))




#3. Define two non square matrices such that they can be mulplied.
A=np.array([[1,2,3],[2,3,4]])
B=np.array([[3,4],[4,5],[5,6]])
print("Two non square matrices")
print(A)
print(B)

#4. Print the shape of the above matrices
print("Shape of the above matrices")
print(np.shape(A))
print(np.shape(B))


#5. Print the product of above two matrices (do so without using the inbuilt functions).
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