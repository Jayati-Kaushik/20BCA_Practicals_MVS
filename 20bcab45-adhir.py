import numpy as np
#1
x = np.array([1,2,3,4,5,6,7])
#2
print(x[1])
print(x[3])
print(x[6])
#3
print(np.shape(x))
#4
x = np.array([[1,2,3,4,5,6,7]]).T
print(x)
#5
M = np.array([[4,6,8,2,6],[12,16,18,20,15],[22,24,23,21,20],[94,96,91,97,99]])
print(M)
#6
print(np.shape(M))
#7
M = np.array([[4,6,8,2,6],[12,16,18,20,15],[22,24,23,21,20],[94,96,91,97,99]])
print(M)
#8
print(M[:,0])
#9
print(M[0])
#10
print(np.array([[0,0,0,0,0,0,0]]*7))
print(np.array([[1,1,1]]*3))