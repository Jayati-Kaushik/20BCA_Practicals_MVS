# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
x = np.array([1,2,3,4,5,6,7])
print(x)

print(x[1])
print(x[3])
print(x[7])#there is no 7th index in this array so we will get an error

print(np.shape(x))

x = np.array([[1,2,3,4,5,6,7,8]])
print(x)
print(np.shape(x))

y = np.array([[1,3,5,7,9],[12,14,16,18,20],[22,21,26,14,21],[72,32,42,31,63]])
print(y)

print("First coloumn")
print(y[:,0:1])

print("First Column")
print(y[0:1])

print(np.array([[0,0,0,0,0,0,0]]*7)) 
print(np.array([[1,1,1]]*3))