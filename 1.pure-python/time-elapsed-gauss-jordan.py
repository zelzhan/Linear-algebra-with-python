import numpy as np
import time


#implementation of Gauss-Jordan Elimination for square matrices
test1 = [[ 0,  2,  1],
         [ 1, -2, -3],
         [-1,  1,  2]]

test2 = [[1, -2, 1],
        [0, 2, -8],
        [-4, 5, 9]]

test3 = [[1, 1, -2, 1, 3, -1],
         [2, -1, 1, 2, 1, -3],
         [1, 3, -3, -1, 2, 1],
         [5, 2, -1, -1, 2, 1],
         [-3, -1, 2, 3, 1, 3],
         [4, 3, 1, -6, -3, -2]]

another = [[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]]

test_vector1 = [-8, 0, 3]
test2_vector = [0, 8, -9]
test3_vector = [4, 20, -15, -3, 16, -27]


numpy_matrix = np.array(test3)
numpy_vector = np.array([test3_vector])
numpy_vector = np.transpose(numpy_vector)

start1 = time.time()
solution = np.linalg.solve(numpy_matrix, numpy_vector)
end1 = time.time()
numpy_time = end1 - start1



#switches the rows in the case when there is a vector with zero in the first place
def switch_the_rows(mat, vector):
    for i in range(len(list_)):
        if mat[i][0] != mat[0][0] and mat[i][0] != 0:
            temp1 = mat[i]
            temp2 = vector[i]
            mat[i] = mat[0]
            vector[i] = vector[0]
            mat[0] = temp1
            vector[0] = temp2
            break
    return mat, vector  

#row reduces the matrix, approaches to lower triangular matrix
#one row operation each time
def lower_row_reduction(mat, pivot_index, vector):
    pivot = mat[pivot_index]
    for i in range(pivot_index, len(mat)):
        if i+1 == len(mat): break
        if mat[i+1][i] != 0:
            piv_multiple = [-x * mat[i+1][pivot_index]/pivot[pivot_index] for x in pivot]
            vect_multiple = -vector[pivot_index]*mat[i+1][pivot_index]/pivot[pivot_index]                        
            mat[i+1][:] = [x+y for x, y in zip(piv_multiple, mat[i+1])]
            vector[i+1] = vect_multiple+vector[i+1]
    return mat, vector

#creation of lower triangular matrix
def lower_triangular(mat, vector):
    for i in range(len(mat)):
        mat, vector = lower_row_reduction(mat, i, vector)
    return mat, vector

#row reduces the matrix, approaches to identity matrix
#one row operation each time
def upper_row_reduction(mat, pivot_index, vector):
    pivot = mat[pivot_index]
    vector[pivot_index] /= pivot[pivot_index] 
    mat[pivot_index][:] = [x/pivot[pivot_index] for x in pivot]
    pivot = mat[pivot_index]
    for i in range(pivot_index, -1, -1):
        if i == 0: break
        if mat[i][i] != 0:
            temp = [-x*mat[i-1][pivot_index] for x in pivot]
            temp_vector = -vector[pivot_index]*mat[i-1][pivot_index] 
            mat[i-1][:] = [y+x for x, y in zip(mat[i-1], temp)]   
            vector[i-1] = temp_vector+vector[i-1]
    return mat, vector

#creation of upper triangular matrix
def upper_triangular(mat, vector):
    for i in range(len(mat)-1, -1, -1):
        mat, vector = upper_row_reduction(mat, i, vector)
    return mat, vector

#merge the functions    
def gauss_jordan(mat, vector):
    if mat[0][0] == 0: mat, vector = switch_the_rows(mat, vector)
    mat, vector = lower_triangular(mat, vector)  
    mat, vector = upper_triangular(mat, vector)
    return mat, vector

start = time.time()
M, sol = gauss_jordan(test3, test3_vector)
end = time.time()

my_time = end - start 

sol = [round(x, 2) for x in sol]                #round each element in the vector up to 2 decimal places
print(sol)

print("Numpy implementation: " + str(numpy_time))
print("My implementation: " + str(my_time))
print("Difference: " + str(my_time - numpy_time))
                
