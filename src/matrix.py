def matrix_multiply(A : In[Array[float]], 
                   B : In[Array[float]], 
                   C : Out[Array[float]], 
                   M : In[int], 
                   N : In[int], 
                   K : In[int]):
    i : int = 0
    while (i < M, max_iter := 1000):
        j : int = 0
        while (j < N, max_iter := 1000):
            sum_val : float = 0.0
            k : int = 0
            while (k < K, max_iter := 1000):
                a_idx : int = i * K + k
                b_idx : int = k * N + j
                sum_val = sum_val + A[a_idx] * B[b_idx]
                k = k + 1
            
            c_idx : int = i * N + j
            C[c_idx] = sum_val
            j = j + 1
        i = i + 1

# # SIMD parallel version for better performance
# @simd
# def matrix_multiply_parallel(A : In[Array[float]], 
#                            B : In[Array[float]], 
#                            C : Out[Array[float]], 
#                            M : In[int], 
#                            N : In[int], 
#                            K : In[int]):
#     thread_idx : int = thread_id()
    
#     # Convert linear thread index to 2D matrix coordinates
#     total_elements : int = M * N
#     if thread_idx < total_elements:
#         i : int = thread_idx / N  # Row index (integer division)
#         j : int = thread_idx - i * N  # Column index
        
#         # Compute C[i][j] = sum over k of A[i][k] * B[k][j]
#         sum_val : float = 0.0
#         k : int = 0
#         while (k < K, max_iter := 1000):
#             a_idx : int = i * K + k
#             b_idx : int = k * N + j
#             sum_val = sum_val + A[a_idx] * B[b_idx]
#             k = k + 1
        
#         c_idx : int = i * N + j
#         C[c_idx] = sum_val

# Vector-matrix multiplication (special case)
def vector_matrix_multiply(v : In[Array[float]], 
                          M : In[Array[float]], 
                          result : Out[Array[float]], 
                          vec_size : In[int], 
                          mat_cols : In[int]):
    j : int = 0
    while (j < mat_cols, max_iter := 1000):
        sum_val : float = 0.0
        i : int = 0
        while (i < vec_size, max_iter := 1000):
            m_idx : int = i * mat_cols + j
            sum_val = sum_val + v[i] * M[m_idx]
            i = i + 1
        result[j] = sum_val
        j = j + 1

# Matrix-vector multiplication
def matrix_vector_multiply(M : In[Array[float]], 
                          v : In[Array[float]], 
                          result : Out[Array[float]], 
                          mat_rows : In[int], 
                          mat_cols : In[int]):
    i : int = 0
    while (i < mat_rows, max_iter := 1000):
        sum_val : float = 0.0
        j : int = 0
        while (j < mat_cols, max_iter := 1000):
            m_idx : int = i * mat_cols + j
            sum_val = sum_val + M[m_idx] * v[j]
            j = j + 1
        result[i] = sum_val
        i = i + 1