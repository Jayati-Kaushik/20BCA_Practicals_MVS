import numpy as np
#1 
x=([1,2,3,4,5,6,7,8,9])
print(x)
#2
print(x[1], x[4], x[6])
#3
print(np.shape(x))
#4
x=np.array([[1,2,3,4,5,6,7,8,9]]).T
print(x)
print(np.shape(x))
#5
z=np.array([[1,3,3,2,1],[55,5,6,7,3],[9,8,7,6,5],[3,4,5,6,7]])
print(z)
#6
print(np.shape(z))
#7
z=np.array([[1,3,3,2,1],[55,5,6,7,3],[9,8,7,6,5],[3,4,5,6,7]]).T
print(z)
#8
print(z[1,1])
#9
print(z[0,:1])
#10
print(np.array([[0,0,0,0,0,0,0]]*7))
print(np.array([[1,1,1]]*3))
