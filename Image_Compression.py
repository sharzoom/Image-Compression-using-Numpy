#Q1
import numpy as np               #imported numpy
A = np.matrix('7 8; 2 5')        #used np.matrix to create a matrix
print("Q1 Answer:", A)           #printed the results

#Q2
from numpy import linalg as LA           #imported module callled linalg(linear algebra)
eigenvalues, eigenvectors = LA.eig(A)    #used LA.eig() to find eigenvalues and eigenvectors
a=[]                                     #created an empty list called 'a'
for i in range (len(eigenvalues)):       #initiated a for loop that will iterated over each eigenvalue
    eigenvalue = eigenvalues[i]          #save the corresponding eigenvalues to the variable 'eigenvalue'
    xi = eigenvectors[:,i]               #this line extracts the i-th eigenvector from the eigenvectors, [:,i] will select all rows and ith column 
    LHS = A.dot(xi)                      #performed matrix multiplication between A and xi and set value to LHS
    RHS = xi.dot(eigenvalue)             #performed matrix multiplication between xi and eigenvalue and set value to RHS
    a.append((xi,eigenvalue,LHS,RHS))    #appended all values to a
print("\n Q2 Answer:", a)                #printed the results
# print(type(a))


#Q3
B = np.matrix('0 1; 1 0')                                                           #used np.matrix to create matrix 'B'
C = np.matrix('2 3 0; 3 4 1; 0 1 9')                                                #used np.matrix to create matrix 'C'
def check_orthogonality(matrix):                                                    #created a function to check if eigenvectors are orthogonal
    eigenvalues, eigenvectors = np.linalg.eig(matrix)                               
    eigenvectors = eigenvectors.T                                                   #using '.T' took transpose of eigenvectors
    product = np.dot(eigenvectors, eigenvectors.T)                                  #performed matrix multiplication between eigenvectors and eigenvectors.T and set value to product
    if np.allclose(product, np.eye(len(eigenvalues))):                              #used an if loop to check if the matrix is identity matrix or not. np.eye(len(eigenvalues)) creates an identity matrix with the same size as the number of eigenvalues and np.allclose will chech if product is close to this identiyty matrix.
        print("True, Eigenvectors are orthogonal, hence the given matrix is symmetric.")  #prints if the matrix is close to identity matrix. ie, symmetric.
    else:
        print("False, Eigenvectors are not orthogonal.")                                   #else prints the matrix is not symmetric.
print("\n Q3 Answer:")                                                              #printed the answers
print("Matrix A:")                                                                  #called the function check_orthogonality( ), and passed matrix A
check_orthogonality(A)                                                              #calling function check_orthogonality( ) to check if the matrix is orthogonal 
print("\nMatrix B:")                                                                #printed the answers
check_orthogonality(B)                                                              #calling function check_orthogonality( ) to check if the matrix is orthogonal 
print("\nMatrix C:")                                                                #printed the answers
check_orthogonality(C)                                                              #calling function check_orthogonality( ) to check if the matrix is orthogonal 

#Q4
eigenvalues, eigenvectors = np.linalg.eig(A)      #used LA.eig() to find eigenvalues and eigenvectors of corresponding matrixes
Λ = np.diag(eigenvalues)                          #np.diag() is a NumPy function that constructs a diagonal matrix of eigenvalues
Q = eigenvectors                                  #assigned eigenvectors to a variable called Q
Qinv = np.linalg.inv(Q)                           #using np.linalg.inv(Q) took inverse of Q
QΛ= np.dot(Q,Λ)                                   #performed matrix multiplication between Q and Λ and set value to QΛ
QΛQinv = np.dot(QΛ, Qinv)                         #performed matrix multiplication between QΛ and Qinv and set value to QΛQinv
print("\n Q4 Answer:")                            #printed the answers
print("\n Matrix Q:", Q)
print("\n Matrix Λ:", Λ)
print("\n Matrix QΛQ^{-1}:", QΛQinv)

#Q5
print("\n Q5 Answer:")
eigenvalues, eigenvectors = np.linalg.eig(C)    #used LA.eig() to find eigenvalues and eigenvectors
Λ = np.diag(eigenvalues)                        #np.diag() is a NumPy function that constructs a diagonal matrix of eigenvalues as like L41
Q = eigenvectors                                #assigned eigenvectors to a variable called Q
print("\nMatrix C:")
check_orthogonality(C)                          #calling function check_orthogonality( ) to check if the matrix is orthogon
print("\nMatrix Q:")
check_orthogonality(Q)                          #calling function check_orthogonality( ) to check if the matrix is orthogonal 
QQT = np.dot(Q, Q.T)                            #performed matrix multiplication between Q and Q.T, which is transpose of Q and set value to QQT
print("\n Matrix QQTranspose:", QQT)            #printed the answer
Qinv = np.linalg.inv(Q)                         #using np.linalg.inv(Q) took inverse of Q and assigned it to Qinv
QQinv = np.dot(Q, Qinv)                         #performed matrix multiplication
print("\n Matrix QQ^{-1}:", QQinv)              #printed the answer
QΛ= np.dot(Q,Λ)                                 #performed matrix multiplication
QΛQT = np.dot(QΛ, Q.T)                          #performed matrix multiplication
print("\n Matrix QΛQTranspose:", QΛQT)          #printed the answer
QΛQinv = np.dot(QΛ, Qinv)                       #performed matrix multiplication
print("\n Matrix QΛQ^{-1}:", QΛQinv)            #printed the answer

#Q6
C = np.matrix('1 2; 6 10; 5 6')                #used np.matrix to create matrix 'C'
U, sigma, Vh = np.linalg.svd(C)                #used np.linalg.svd() to do Singular value decomposition on matrix C. It returns three matrices: U (left singular vectors), sigma (singular values), and Vh (conjugate transpose of the right singular vectors).
m, n = C.shape                                 #using C.shape it helps to assign no: of rows and columns to m & n respectevely
complete_sigma = np.zeros((m, n))              #using np.zeros((m, n)) initiated a new matrix with zero that has same dimension of matrix c 
complete_sigma[:n, :n] = np.diag(sigma)        #this line construct a new matrix that adds values in sigma as diagonal using np.diag(sigma)
USVtranspose = U @ complete_sigma @ Vh         #computed UsigmaVh using '@' which is used to perform matrix multiplication
V = Vh.T                                       #took transpose of Vh and assigned it to V
print("\n Q6 Answer:")                         #print matrices
print("Matrix U:", U)
print("\nMatrix Σ:", complete_sigma)
print("\nMatrix V:", V)
print("\nMatrix UΣV^T:", USVtranspose)

#Q7 

from PIL import Image                                                                                           #install the PIL package if not previously installed
image_np = np.asarray(Image.open("C:\\Users\\Admin\\Desktop\\Spring sem\\DL\\mountain_grayscale.jpg"))          #importing image in the variable images_np
U, sigma, Vh = np.linalg.svd(image_np)                                                                          #used np.linalg.svd() to do Singular value decomposition on the given image. It returns three matrices: U (left singular vectors), sigma (singular values), and Vh (conjugate transpose of the right singular vectors).

compression_ratio = 2                                                                                           #set a varaible called 'compression_ratio' and assigned value 2 to it
uncompressed_image_size = image_np.shape[0] * image_np.shape[1]                                                 #multiplied dimensions of the image  to find uncompressed image size

denominator = compression_ratio * (1 + image_np.shape[0] + image_np.shape[1])                                   #To calculate number of singular values used first calculated the denominator which is compressin ratio * (1 + height of image + width of image)
NoOfSVused = uncompressed_image_size / denominator                                                              #calculated the number of singular value used 
compressed_image_size = NoOfSVused * (1 + image_np.shape[0] + image_np.shape[1])                                #using the NoOfSVused calculated ompressed image size

U_cr = U[:,:int(NoOfSVused)]                                                                                    #extracts the first NoOfSVused columns of the left singular vectors matrix U
sigma_cr = sigma[:int(NoOfSVused)]                                                                              #extracts the first NoOfSVused singular values from the array sigma. 
Vh_cr = Vh[:int(NoOfSVused),:]                                                                                  #extracts the first NoOfSVused rows of the transpose of the right singular vectors matrix Vh.
compressed_image_np = np.dot(U_cr * sigma_cr, Vh_cr)                                                            #reconstructs the compressed image matrix by multiplying U, sigma and Vh 
print("\n Q7 Answer:")  
print("Dimension of the final compressed image:", compressed_image_np.shape)                                    #print the results
print("Number of singular values used:", int(NoOfSVused))

compressed_mountain_image = Image.fromarray(compressed_image_np.astype(np.uint8), mode="L")                     
compressed_mountain_image.save("C:\\Users\\Admin\\Desktop\\Spring sem\\DL\\mountain_grayscale_compressed.jpg")  #Save the compressed image to a file to the provided path
orginal_mountain_image = Image.fromarray(image_np, mode="L")
orginal_mountain_image.save("C:\\Users\\Admin\\Desktop\\Spring sem\\DL\\mountain_grayscale_orginal.jpg")        #Save the compressed image to a file to the provided path

print("Uncompressed_image_size:", uncompressed_image_size)                                                      #print the results
print("Compressed_image_size:", compressed_image_size)
