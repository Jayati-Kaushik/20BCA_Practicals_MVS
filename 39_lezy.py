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
M = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]])
print(M)
#6
print(np.shape(M))
#7
M = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]]).T
print(M)
#9
print(M[0])
#8
print(M[:,0])
#10
print(np.array([[0,0,0,0,0,0,0]]*7))
print(np.array([[1,1,1]]*3))
