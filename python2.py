# -*- coding: utf-8 -*-
"""
Created on Tue Oct  12 12:49:44 2021
@author: Admin
"""

import numpy as np
#1. 
x=np.array([[1,1,1,1,1,1],[2,2,2,2,2,2],[3,3,3,3,3,3],[4,4,4,4,4,4],[5,5,5,5,5,5],[6,6,6,6,6,6]])
print(x)

#2. 
X=np.array([[1,1,1,1,1,1],[2,2,2,2,2,2],[3,3,3,3,3,3],[4,4,4,4,4,4],[5,5,5,5,5,5],[6,6,6,6,6,6]]).T
print(X)

#3.
a=np.array([[1,2,6],[8,2,9]])
b=np.array([[2,5],[8,1],[2,6]])
print(a)
print(b)

#4. 
print(a.shape)
print(b.shape)

#5. 
Z=np.array([np.zeros(3)]*3)
for i in range(len(a)):
    for j in range(len(b[1])):
        for k in range(len(b)):
            Z[i][j] += a[i][k] * b[k][j]
print(Z)

#6. 
p=np.array([[5,3,9],[9,2,4],[9,4,7]])
q=np.array([[9,3,6],[5,2,8],[8,1,2]])
print(p+q)

#7. 
A=np.array([[7,3],[9,2]])
print(A)

#8. 
print(A.T)

#9. 
I=np.array([[1,0],[0,1]])
print(I)

#10. 
X=A@I
print("A.I = ",X)
Y=I@A
print("I.A = ",Y)
print(" Therefore, A.I = I.A") 

#11.       
m=np.array([[4,1],[7,2]])
print(m)

#12. 
print(A@m)

#13. 
print(np.multiply(A,m))    

#14. 
c=np.linalg.det(A)
print("The determinant of A is",c)
if c!=0:
    print("The inverse is ",np.linalg.inv(A))
else:
    print("Inserve does not exist")