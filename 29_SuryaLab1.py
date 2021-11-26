# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
#1 
x=(3,5,8,1,4,9,78)
print(x)
# 2 
print(x[1])
print(x[3])
print(x[6])
# 3
print(np.shape(x))
# 4
X=np.array([[3,5,8,1,4,9,78]]).T
print(x)
print(np.shape(X))
#5 
z=np.array([[1,1,2,3,2],[22,3,55,1,5],[3,7,8,66,8],[98,2,1,0,7]])
print(z)
#6
print(np.shape(z))
#7
Z=np.array([[1,1,2,3,2],[22,3,55,1,5],[3,7,8,66,8],[98,2,1,0,7]]).T
print(z)
#8
print(z[1,1])
#9
#10
print(np.array([[0,0,0,0,0,0,0]]*7))
print(np.array([[1,1,1]]*3))      
