# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 16:39:49 2022

@author: prajw
"""

import numpy as np

a=np.array([[1,2,3,4,5,6],[2,4,6,8,10,12],[3,6,9,12,15,18],[4,8,12,16,20,24],[5,10,15,20,25,30],[6,12,18,24,30,36]])
print(a)

b=np.array([[1,2,3,4,5,6],[2,4,6,8,10,12],[3,6,9,12,15,18],[4,8,12,16,20,24],[5,10,15,20,25,30],[6,12,18,24,30,36]]).T
print(b)

c=np.array([[1,2,3],[4,5,6]])
d=np.array([[2,4],[3,6],[4,8]])
print(c)
print(d)

print(c.shape)
print(d.shape)

Z=np.array([np.zeros(3)]*3)
for i in range(len(c)):
    for j in range(len(d[1])):
        for k in range(len(d)):
            Z[i][j] += c[i][k] * d[k][j]
print(Z)

p=np.array([[3,6,9],[2,4,6],[4,8,7]])
q=np.array([[9,8,4],[5,2,8],[7,5,1]])
print(p+q)

f=np.array([[1,3],[4,2]])
print(f)

g=np.array([[1,3],[4,2]]).T
print(g)

I=np.array([[1,0],[0,1]])
print(I)

X=f@I
print("A.I = ",X)
Y=I@f
print("I.A = ",Y)
print(" Therefore, A.I = I.A") 

h=np.array([[4,8],[7,5]])
print(h)

print(h*f)

print(np.multiply(f,h)) 

s=np.linalg.det(f)
print("The determinant of A is",s)
if s!=0:
    print("The inverse is ",np.linalg.inv(f))
else:
    print("Inserve does not exist")