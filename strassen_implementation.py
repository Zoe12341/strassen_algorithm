
#Name:       - Zoe Buck
#References: - https://en.wikipedia.org/wiki/Strassen_algorithm, https://www.cuemath.com/algebra/multiplication-of-matrices/ 

#brute force matrix multiplication
def multiply_bf(A, B):
    """
    Multiplies two matrices A and B using brute force. The number of columns in A must be equal to the number of rows in B.
    
    :param A: (array of arrays of ints or floats) Represents a matrix of size m x n
    :param B: (array of arrays of ints or floats) Represents a second matrix of size n x p 
        
    :return C: (array of arrays of ints or floats) The resulting matrix of size m x p from A multiplied by B.

    >>> A = [[-1,4],[2,3]]
    >>> B = [[9,-3], [6,1]]
    >>> multiply_bf(A, B)
    [[15, 7],[36, -3]] 
    """
    num_Acols = len(A[0])
    num_Bcols = len(B[0])

    C = [[0]*num_Bcols for i in range(len(A))]

    for rowA in range(len(A)):
        for colA in range(num_Acols):
            for colB in range(num_Bcols):
                C[rowA][colB] += B[colA][colB] * A[rowA][colA]
    return C


def add_matrices(A, B):
    """
    Given two matrices A and B of size nxn, matrix B and matrix A are added and the result is returned. The element at each index i,j in B is added to the element at the same i, j index in A. 
    
    :param A: (array of arrays of ints or floats) Represents a matrix of size n x m
    :param B: (array of arrays of ints or floats) Represents a second matrix of size n x m
        
    :return result: (array of arrays of ints or floats) The n x m matrix resulting from matrix A + matrix B

    >>> A = [[1,2], [3,4]]
    >>> B = [[5,0], [7,2]]
    >>> add_matrices(A, B)
    [[6, 2],[10, 6]]  
    """
    n = len(A)
    result = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            result[i][j] = A[i][j] + B[i][j]
    return result


def subtract_matrices(A, B):
    """
    Given two matrices A and B of size nxn, matrix B is subtracted from matrix A and the result is returned. The element at each index i,j in B is subtracted from the element at the same i, j index in A. 
    
    :param A: (array of arrays of ints or floats) Represents a matrix of size n x m
    :param B: (array of arrays of ints or floats) Represents a second matrix of size n x m
        
    :return result: (array of arrays of ints or floats) The n x m matrix resulting from matrix A - matrix B

    >>> A = [[1,2], [3,4]]
    >>> B = [[5,0], [7,2]]
    >>> subtract_matrices(A, B)
    [[-4, 2],[-4, 2]]  
    """
    n = len(A)
    result = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            result[i][j] = A[i][j] - B[i][j]
    return result


def strassen(A, B):
    """
    Multiplies two matrices A and B using Strassen's algorithm. A and B must both be of size n x n. 
    
    :param A: (array of arrays of ints or floats) Represents a matrix of size n x n
    :param B: (array of arrays of ints or floats) Represents a second matrix of size n x n 
        
    :return C: (array of arrays of ints or floats) The resulting matrix of size n x n from A multiplied by B.

    >>> A = [[-1,4],[2,3]]

    >>> B = [[9,-3], [6,1]]
    >>> strassen(A, B)
    [[15, 7],[36, -3]] 
    """
    n = len(A)
    
    # Base case: 1x1 matrix
    if n == 1:
        return [[A[0][0] * B[0][0]]]
    
    # Initialize submatrices of size n/2 x n/2
    middle = n // 2
    sub_a = [[0 for _ in range(middle)] for _ in range(middle)]
    sub_b = [[0 for _ in range(middle)] for _ in range(middle)]
    sub_c = [[0 for _ in range(middle)] for _ in range(middle)]
    sub_d = [[0 for _ in range(middle)] for _ in range(middle)]
    sub_e = [[0 for _ in range(middle)] for _ in range(middle)]
    sub_f = [[0 for _ in range(middle)] for _ in range(middle)]
    sub_g = [[0 for _ in range(middle)] for _ in range(middle)]
    sub_h = [[0 for _ in range(middle)] for _ in range(middle)]

    # Fill values into the submatrices
    for i in range(middle):
        for j in range(middle):
            sub_a[i][j] = A[i][j]
            sub_b[i][j] = A[i][j + middle]
            sub_c[i][j] = A[middle + i][j]
            sub_d[i][j] = A[middle + i][j + middle]
            
            sub_e[i][j] = B[i][j]
            sub_f[i][j] = B[i][j + middle]
            sub_g[i][j] = B[middle + i][j]
            sub_h[i][j] = B[middle + i][j + middle]

    # Calculate M1 to M7 using recursion and helper functions
    M1 = strassen(add_matrices(sub_a, sub_d), add_matrices(sub_e, sub_h))
    M2 = strassen(add_matrices(sub_c, sub_d), sub_e)
    M3 = strassen(sub_a, subtract_matrices(sub_f, sub_h))
    M4 = strassen(sub_d, subtract_matrices(sub_g, sub_e))
    M5 = strassen(add_matrices(sub_a, sub_b), sub_h)
    M6 = strassen(subtract_matrices(sub_c, sub_a), add_matrices(sub_e, sub_f))
    M7 = strassen(subtract_matrices(sub_b, sub_d), add_matrices(sub_g, sub_h))
    
    # Combine M1 to M7 into resulting matrix C
    C = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(middle):
        for j in range(middle):
            C[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j]
            C[i][j + middle] = M3[i][j] + M5[i][j]
            C[middle + i][j] = M2[i][j] + M4[i][j]
            C[middle + i][j + middle] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j]
    
    return C


def main():
    ''' Main function used to test results of matrix multiplication functions... add more test cases '''
    # Example usage:
    A = [[1, 2],
        [3, 4]]

    B = [[5, 6],
        [7, 8]]

    A2 = [[-1,4],
        [2,3]]

    B2 = [[9,-3],
        [6,1]]


    result_strassen2 = strassen(A2, B2)
    result_bf2 = multiply_bf(A2, B2)
    correct_result2 = [[15, 7],[36, -3]] 

    if result_strassen2 == result_bf2 == correct_result2:
        return True
    else:
        return False


    #uncomment code below to print result_strassen2 matrix
    # print("Result of A2 x B2 using Strassen's algorithm:")
    # for row in result_strassen2:
    #     print(row)






if __name__ == "__main__":
    main()

