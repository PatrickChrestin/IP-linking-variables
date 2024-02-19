import matplotlib.pyplot as plt
import numpy as np
import math
from gurobipy import *
import staircasedetection_sparse 
from scipy.sparse import csr_matrix

def count_nonzero_entries_per_row(matrix):
    num_rows = matrix.shape[0]
    non_zero_counts = []
    for i in range(num_rows):
        row = matrix.getrow(i)
        non_zero_count = row.nnz
        non_zero_counts.append(non_zero_count)
    return non_zero_counts

def visualize(coef_matrix, sorted_vars_dict, sorted_constraints_dict):
    print(sorted_vars_dict)
    print(sorted_constraints_dict)
    A_stair = csr_matrix(coef_matrix.shape)
    # initialize index arrays of matrix A_stair
    sorted_vars_array = np.concatenate(list(sorted_vars_dict.values()))
    sorted_constraints_array = np.concatenate(list(sorted_constraints_dict.values()))
    rows, cols = coef_matrix.nonzero()
    for rows, cols in zip(rows, cols):
        A_stair[np.where(sorted_constraints_array == rows+1)[0][0], np.where(sorted_vars_array == cols+1)[0][0]] = 1
    A_stair_dense = A_stair.todense() 

    A_blocks = np.zeros(A_stair_dense.shape)
    A_linking = np.zeros(A_stair_dense.shape)
    index_row = 0
    index_col = 0
    cur_block = 1
    while (cur_block <= (2*k-1)): 
        print(f"cur_block: {cur_block}")
        print(f"index_row: {index_row}, index_col: {index_col}")
        block_length = 0
        block_height = 0
        if cur_block %2 == 1: #case separation: even are linking blocks, uneven are independent blocks
            block_length = len(sorted_vars_dict[str(math.ceil(cur_block/2))])
            block_height = len(sorted_constraints_dict[str(math.ceil(cur_block/2))])
        else:
            block_length = len(sorted_vars_dict[str(math.ceil(cur_block/2))+","+str(math.ceil(cur_block/2)+1)])
            block_height = len(sorted_constraints_dict[str(math.ceil(cur_block/2))])+len(sorted_constraints_dict[str(math.ceil(cur_block/2)+1)])
        print(f"block_length: {block_length}, block_height: {block_height}")
        if cur_block %2 == 1:
            for j in range(0,block_length):
                for i in range(0,block_height):
                    A_blocks[index_row+i,index_col+j] = 1
            print(A_blocks)
        else: 
            for j in range(0,block_length):
                for i in range(0,block_height):
                    A_linking[index_row+i,index_col+j] = 1 
            print(A_linking)
        # if cur_block < k:
        if cur_block %2 == 0:
            index_row += len(sorted_constraints_dict[str(math.ceil((cur_block-1)/2))])
        index_col += block_length 
        
        cur_block += 1

    plt.imshow(A_blocks, cmap='Blues', alpha=0.8)
    plt.imshow(A_linking, cmap='Oranges', alpha=0.8)
    # add a second layer to color all non-zero entries in the matrix in light blue
    plt.imshow(A_stair_dense, cmap='Greys', alpha=0.5)

    plt.savefig('staircase_vpm2.png')
    plt.show()

vpm2 = gurobipy.read("vpm2/vpm2.mps")

# input coefficient matrix and parameters
A = vpm2.getA()
coef_matrix_binary = csr_matrix((A.data != 0, A.indices, A.indptr), shape=A.shape)
# A_dense = coef_matrix_binary.todense() 
# plt.imshow(A_dense, cmap='Greys')
# plt.savefig('A_blkwht.png')
# plt.show()

r_sum = count_nonzero_entries_per_row(coef_matrix_binary)
N = range(1,coef_matrix_binary.shape[1]+1)
M = range(1,coef_matrix_binary.shape[0]+1)
k = 8
model, sorted_vars_dict, sorted_constraints_dict = staircasedetection_sparse.solve(N, M, k, coef_matrix_binary, r_sum, True)
visualize(coef_matrix_binary, sorted_vars_dict, sorted_constraints_dict)

model.write('staircasedetection.lp')
if model.Status == GRB.INFEASIBLE:
    model.computeIIS()
    model.write('iismodel.ilp')