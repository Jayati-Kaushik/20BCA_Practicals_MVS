# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 11:30:41 2021

@author: Sohan Immanuel
"""

import numpy as np

#CREATE AND PRINT 7 TUPLE ID ARRAY
x = np.array([1,2,3,4,5,6,7])

#PRINT 2ND,4TH AND 7th array element
print(x[1]) 
print(x[3])
print(x[6])

#PRINT THE SHAPE OF YOUR ARRAY
print(x.shape)

#PRINT THE TRANSPOSE OF THE ARRAY
print((x).T)


import numpy as np

#CREATE A 4X5 VECTOR MATRIX
m1 = np.array([[1,2,3,4,],[1,1,1,3],[1,4,5,3],[1,5,7,2],[1,9,7,8]])
print(m1)

#PRINT THE SHAPE OF THE MATRIX
print(m1.shape)

#PRINT THE TRANSPOSE OF THE MATRIX
print(m1.T)

#PRINT THE FIRST COLUMN OF THE MATRIX 
print(m1[0,0],m1[1,0],m1[2,0],m1[3,0])

#PRINT THE FIRST ROW OF THE MATRIX 
print(m1[0,0],m1[0,1],m1[0,2],m1[0,3])

#CREATE 7 DIMENSIONAL ARRAY WITH ONLY ZERO'S
print(np.zeros(7))

#CREATE A 3 DIMENSIONAL ARRAY WITH ONLY ONE'S
print(np.ones(3))

