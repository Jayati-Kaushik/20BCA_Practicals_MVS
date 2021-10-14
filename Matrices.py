import numpy as np
# Define and print a 6 dimentional vector
A=np.array([[1,2,3,4,5,6],
            [7,4,2,8,9,7],
            [6,1,2,6,7,1],
            [7,1,2,2,3,6],
            [1,4,6,8,3,2],
            [3,4,3,1,9,7]])
print("6 dimentional vector is\n",A)

# Transpose of the Vector
print("Transpose :\n",np.transpose(A))

# 2 non square matrices that can be multipled
A=np.array([[3,4,5],[4,5,6]])
B=np.array([[3,4],[4,5],[5,6]])
print("Two non square matrices")
print(A)
print(B)

# Shape of the above matrice
print("Shape of the above matrices")
print(np.shape(A))
print(np.shape(B))

# Product of the matrices
print("Product of above matrices")
result=[[0,0],
        [0,0]]
for i in range(len(A)):
    for j in range(len(B[0])):
        for k in range(len(B)):
            result[i][j] = result[i][j]+A[i][k]*B[k][j]
for r in result:
    print (r)

# 2 non square matrices and their sum
print("Two non square matrices of same order")
X=np.array([[3,2,1],[2,6,8]])
Y=np.array([[4,5,6],[8,12,9]])
print(X)
print(Y)
print("Sum of above matrices")
result=[[0,0,0],
        [0,0,0]]
for i in range(len(X)):
    for j in range(len(Y[0])):
        result[i][j]=X[i][j]+Y[i][j]
for r in result:
    print (r)

# Square matrix A
A=np.array([[9,16,13],[6,9,7],[12,7,3]])
print("Square matrix A")
print(A)

# Transpose of A
print("Transpose of above matrix\n",np.transpose(A))

# Identity matrix
print("Identity matrix of same order")
I=np.identity(3)
print(I)

# Verify A.I = I.A for matrix multiplication.
A=np.array([[10,11,12],[5,9,2],[15,8,4]])
I=np.identity(3)
print("A*I")
print(np.dot(A,I))
print("I*A")
print(np.dot(I,A))
if(np.dot(A,I).all()==np.dot(I,A).all()):
    print("Verified")
else:
    print("Not verified")

#anothersquare matrix of the same order
print("Square matrix B of same order as A")
B=np.array([[9,6,8],[5,3,6],[3,6,7]])
print(B)

# matrix multiplication
print("Product of A and B")
print(np.dot(A,B))

# element wise multiplication
print("Product of above matrices by element wise multiplication")
print(np.multiply(A,B))

# Calculate and print the inverse of A. (Use linalg)
# determinant = 0
A=([[1,2,3],[3,5,7],[3,4,5]])
D=np.linalg.det(A)
if (D==0):
    print("Inverse doesnt exist")
else:
    print("Inverse exist")
    print("Inverse of A is")
    print(np.linalg.inv(A))

# Determinant != 0
A=([[2,3,4],[3,4,5],[6,8,9]])
D=np.linalg.det(A)
if (D==0):
    print("Inverse doesnt exist")
else:
    print("Inverse exist")
    print("Inverse of A is")
    print(np.linalg.inv(A))