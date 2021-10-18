import numpy as np
#6 dimentional vector
x=np.array([[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18],[19,20,21,22,23,24],[25,26,27,28,29,30],[31,32,33,34,35,36]])
print(x)

#transpose
print(x.T)

#two non square matrices
a=np.array([[1,2,3],[4,5,6]])
b=np.array([[3,5],[7,9],[1,8]])
print(a)
print(b)
print(a.shape)
print(b.shape)

#Product of the matrices
res=np.array([[0,0]]*2)
for i in range(len(a)):
    for j in range(len(b[1])):
        for k in range(len(b)):
            res[i][j] += a[i][k] * b[k][j]
print(res)

#Define two non square matrices of same order and print their sum.
s=np.array([[6,3],[1,2],[9,7]])
t=np.array([[7,2],[8,1],[4,0]])
print(s+t)

#Define a square matrix A.
A=np.array([[11,12],[23,24]])
print(A)

#transpose of A.
print(A.T)

#Print the identity matrix of the above order I.
I=np.array([[1,0],[0,1]])
print(I)

#Verify A.I = I.A for matrix multiplication.
X=A@I
print("A.I = ",X)
Y=I@A
print("I.A = ",Y)
print(" Therefore, A.I = I.A") 

#Define another square matrix of the same order as A.       
u=np.array([[2,1],[9,6]])
print(u)

#Print the product of the matrices as matrix multiplication
print(A@u)

#Print the product of the matrices by element wise multiplication
print(np.multiply(A,u))    

#Inverse of A.det=np.linalg.det(A)
print("The determinant of A is",det)
if det!=0:
    print("The inverse is ",np.linalg.inv(A))
else:
    print("Inverse does not exist")
