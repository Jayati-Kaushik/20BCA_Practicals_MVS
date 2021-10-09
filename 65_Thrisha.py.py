import numpy as np

x = np.array([1,2,3,4,5,6,7])
print(x)

print(x[1])
print(x[3])
print(x[6])

print(np.shape(x))

x = np.array([[1,2,3,4,5,6,7,8]])
print(x)
print(np.shape(x))

y = np.array([[1,3,5,7,9],[12,14,16,18,20],[22,21,26,14,21],[72,32,42,31,63]])
print(y)

print(np.shape(y))

Y=np.array([[1,3,5,7,9],[12,14,16,18,20],[22,21,26,14,21],[72,32,42,31,63]]).T
print(Y)

print("First Column")
print(Y[:,0:1])

print("First row")
print(Y[0:1])

print(np.array([[0,0,0,0,0,0,0]]*7))
print(np.array([[1,1,1]]*3))