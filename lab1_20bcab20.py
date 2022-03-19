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
M = np.array([[3,9,5,2,1],[11,19,16,13,17],[29,26,21,28,24],[91,99,94,92,97]])
print(M)
#6
print(np.shape(M))
#7
M = np.array([[3,9,5,2,1],[11,19,16,13,17],[29,26,21,28,24],[91,99,94,92,97]]).T
print(M)
#8
print(M[:,0])
#9
print(M[0])
#10
print(np.array([[0,0,0,0,0,0,0,]]*7))
print(np.array([[1,1,1]]*3))
