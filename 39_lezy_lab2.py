import numpy as np 
#1.Define and print a 6 dimensional vector
vc = np.arange(1,37).reshape(6,6)
print(vc)

#2. Print the transpose of the above vector
print(vc.T)

#3.Define two non square matrices such that they can be mulplied.
M = np.array([[1,2,3],[4,5,6]])
print(M)
N = np.array([[9,8],[7,6],[5,4]])
print(N)

#4.Print the shape of the above matrices
print(np.shape(M))
print(np.shape(N))

#5. Print the product of above two matrices (do so without using the inbuilt functions).
result = [[0,0],[0,0]]

for i in range(len(M)):
    for j in range(len(N[0])):
        for k in range(len(N)):
           result[i][j] = result[i][j]+(M[i][k]*N[k][j])
        
for r in result:       
    print(r)
    
#6.Define two non square matrices of same order and print their sum.    
A = np.array([[1,2],[3,4],[5,6]])
print(A)
B = np.array([[7,8],[9,10],[11,12]])
print(B)
print(A+B)

#7.Define a square matrix A
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(A)

#8.Print the transpose of A
print(A.T)

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
B = np.array([[11,12,13],[14,15,16],[17,18,19]])
print(B)

#12.Print the product of the matrices as matrix multiplication
print(A@B)
  
#13.Print the product of the matrices by element wise multiplication
print(np.multiply(A,B))

#14.Calculate and print the inverse of A. (Use linalg)
invrsc = np.linalg.det(A)
print(invrsc)

if invrsc == 0:
    print("Inverse does not exist")
else:
    print("Inverse exist")
print(np.linalg.inv(A))