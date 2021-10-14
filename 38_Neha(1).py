# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

#1
x=np.array([[1,2,3,4,5,6]])
print(x)

#2
print(x.T)

#3
a=np.array([[1,2]])
b=np.array([[3],[2]])
print(a)
print(b)

#4
print(a.shape)
print(b.shape)

#5
p=np.array([[0]])
for i in range(len(a)):
    for j in range(len(b[0])):
        for k in range(len(b)):
            p[i][j]+=a[i][k]*b[k][j]
for r in p:
    print(r)
    
#6
a=np.array([[1,2,3,4],[5,6,7,8]])
b=np.array([[1,3,5,7],[8,7,6,5]])
print(a)
print(b)
print(a+b)

#7
A=np.array([[2,4,6],[1,3,5],[4,5,6]])
print(A)

#8
print(A.T)

#9
I=np.identity(3,int)
print(I)

#10
print(np.dot(A,I))
print(np.dot(I,A))

#11
B=np.array([[1,2,3],[2,3,3],[4,5,6]])
print(B)

#12
print(np.dot(A,B))

#13
print (4*B)
print(A*B)

#14
DET =np.linalg.det(A)
print(DET)
if DET!=0:
    print('inverse exists')
    print(np.linalg.inv(A))
else:
    print('inverse do not exists') 