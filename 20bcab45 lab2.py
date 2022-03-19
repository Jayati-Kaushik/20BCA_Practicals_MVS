
import numpy as np

#1.Define and print a 6 dimensional vector
X = np.array([[2,3,4,5,6,7],[4,7,3,6,1,9],[7,1,4,3,7,9],[2,8,6,1,7,9],[1,5,7,2,8,4],[1,6,8,3,5,7]])
vector1 = np.array(X)
print(vector1)

#2. Print the transpose of the above vector
vector1 = np.transpose(X)
print(vector1)

#3.Define two non square matrices such that they can be mulplied.
M = np.array([[2,5,8],[6,7,1]])
print(M)
N = np.array([[9,4],[4,7],[3,4]])
print(N)

#4.Print the shape of the above matrices
print(np.shape(M))
print(np.shape(N))

#5.Print the product of above two matrices 
result = [[0,0],[0,0]]

for i in range(len(M)):
    for j in range(len(N[0])):
        for k in range(len(N)):
           result[i][j] = result[i][j]+(M[i][k]*N[k][j])
        
for r in result:       
    print(r)

#6.Define two non square matrices of same order and print their sum.
result = [[0,0],[0,0],[0,0]]
Y = np.array([[2,6],[4,8],[5,8]])
print(Y)
Z = np.array([[6,9],[9,1],[2,4]])
print(Z)

for i in range(len(Y)):
    for j in range(len(Y[0])):
        result[i][j] = Y[i][j] + Z[i][j]
        
for r in result:
    print(r)
    
#7.Define a square matrix A
A = np.array([[2,7,4],[8,3,5],[2,4,6]])
print(A)

#8.Print the transpose of A
mat = np.transpose(A)
print(mat)

#9.Print the identity matrix of the above order I.
I = np.identity(3)
print(I)

#10.Verify A.I = I.A for matrix multiplication
result = [[0,0,0],[0,0,0],[0,0,0]]
result= np.dot(A,I)
for r in result:
    print('A.I =',r)
   
    
result = np.dot(I,A)
for r in result:
    print('I.A =',r)
print(" Therefore, A.I = I.A")

    

#11.Define another square matrix of the same order as A
C = np.array([[2,4,6],[5,1,8],[6,8,2]])
print(C)

#12.Print the product of the matrices as matrix multiplication
result = [[0,0,0],[0,0,0],[0,0,0]]
for i in range(len(A)):
     for j in range(len(I)):
        for k in range(len(I)):
            result[i][j] = result[i][j] + (A[i][k]*I[k][j])
for r in result:
    print(r)

result = [[0,0,0],[0,0,0],[0,0,0]]
for i in range(len(I)):
    for j in range(len(A)):
        for k in range(len(A)):
            result[i][j] = result[i][j] + (I[i][k] * A[k][j])
for r in result:
    print(r)
 
#13.Print the product of the matrices by element wise multiplication
result = [[0,0,0],[0,0,0],[0,0,0]]
result = np.multiply(A,I)
for r in result:
    print(r)
    
#14.Calculate and print the inverse of A. (Use linalg)
D = np.linalg.det(A)
print(D)

if D == 0:
    print("Inverse does not exist")
else:
    print("Inverse exist")
print(np.linalg.inv(A))
