# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 19:26:22 2022

@author: musai
"""
import numpy as np

from numpy import linalg 

 
 

#6D vector 

v = np.array([[1,2,3,4,5,6]]) 

print(v) 

print("\n") 

 
 

#tranpose 

print(v.T) 

print("\n") 

 
 

#2 non-square matrix that can be multiplied 

m1=np.array([[1,0,2],[0,1,3]]) 

m2=np.array([[1,2],[2,1],[3,4]]) 

 
 

#matrix shape 

print(m1.shape) 

print(m2.shape) 

print("\n") 

 
 

# matrix product 

print(np.matmul(m1,m2)) 

#z=np.array([np.zeros(2)]*2) 

#for i in range(len(m1)): 

# for j in range(len(m2[1])): 

# for k in range(len(m2)): 

# z[i][j] += m1[i][j] * m2[i][j] 

#print(z) 

print("\n") 

 
 

# sum: two non square matrices of same order 

x=np.array([[1,2,1],[2,1,2]]) 

y=np.array([[1,1,1],[2,2,3]]) 

print(x+y) 

print("\n") 

 
 

# Define a square matrix A. 

a=np.matrix([[1,2],[3,4]]) 

print("\n") 

 
 

# Print the identity matrix of the above order I. 

i=np.array([[1,0],[0,1]]) 

print(i) 

print("\n") 

 
 

# Verify A.I = I.A for matrix multiplication. 

print("A.I= ",a@i) 

print("I.A= ",i@a) 

print("A.I = I.A") 

 
 

# Define another square matrix of the same order as A 

a1=np.array([[4,5],[6,7]]) 

print("\n") 

 
 

# Print the product of the matrices as matrix multiplication 

print(a@a1) 

print("\n") 

 
 

# Print the product of the matrices by element wise multiplication 

print(np.multiply(a,a1)) 

print("\n") 

 
 

# Calculate and print the inverse of A. (Use linalg)Check if determinant is 0 Use if else statement to calculate inverse only when determinant isnon 0 

d=np.linalg.det(a) 

print("Determinnt: ",d) 

if d!=0: 

    print("Inverse: ", np.linalg.inv(a)) 

else: 

    print("Inverse does not exist") 

 