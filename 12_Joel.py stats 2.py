# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 16:01:03 2021

@author: JOEL

#importing numpy
import numpy as np
#defining array
a1 = np.array(([1,2,3,4,5,6]))
#printing Transpose
print(a1.T)
#defining matrices to be multiplied
a= np.array(([2,3],[5,0],[6,9]))
b= np.array(([3,1,4],[2,0,5]))
print(a)
print(b)
#printing shape of marices
print(np.shape(a))
print(np.shape(b))
#method 1 to multiply
a4= np.dot(a,b)
dA= np.linalg.det(a4)
print(dA)
print(a4)
#method 2:
#Calculates number of rows and columns present in first matrix  
row1 = len(a);  
col1 = len(a[0]);  

#Calculates number of rows and columns present in second matrix  
row2 = len(b);  
col2 = len(b[0]);  

#number of columns in the first matrix must be equal to the number of rows in the second matrix  

if(col1 != row2):  
    print("Matrices cannot be multiplied");  

else:  

    prod = [[0]*row1 for i in range(col2)];  

    #Performs product of matrices a and b.  
    for i in range(0, row1):  
        for j in range(0, col2):  
            for k in range(0, row2):  
                prod[i][j] = prod[i][j] + a[i][k] * b[k][j];   

    print("Product of two matrices: ");  
    for i in range(0, row1):  
        for j in range(0, col2):  
            print(prod[i][j]),  

        print(" ");  
#adding matrices
a5=np.array(([2,0],[7,4]))
a6=np.array(([1,3],[9,0]))
print(a5+a6)
#defining sq matrix
A=np.array(([6,2,4],[1,7,5],[4,7,1]))
#printing transpose
print(A.T)
#finding identity matrix
I=np.identity(3)
print(I)
#fining AI,IA and if both are equal
a8=np.dot(A,I)
a9=np.dot(I,A)
print(a8)
print(a9)
#verifying if AI and IA are equal
compare=(a8 == a9)
eq= compare.all()
print(eq)
#other matrix of same order as A
B=np.array(([2,2,1],[1,3,1],[5,7,5]))
#product of matrices as matrix multiplication
V= np.dot(A,B)
dV= np.linalg.det(a4)
print(dV)
print(V)
#Element wise multiplication
C=np.multiply(A,B)
print(C)
#calculate and inverse of A
print((np.linalg.inv(A))
      
