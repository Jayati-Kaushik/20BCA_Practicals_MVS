#1
import numpy as np
t=(1,2,3,4,5,6,7,)
print(t)
#2
print(t[1])
print(t[3])
print(t[6])
#3
print(np.shape(t))
#4
A=np.array([[6,9,12,15,18,4,3]]).T
print(A)
print(np.shape(A))
#5
s=np.array([[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3],[4,4,4,4,4]])
print(s)
#6
print(np.shape(s))
#7
S=np.array([[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3],[4,4,4,4,4]]).T
print(S)
#8
print(s[1:])
print(s[ :1])

print(np.array([[0,0,0,0,0,0,0]]*7))
print(np.array([[1,1,1]]*3))