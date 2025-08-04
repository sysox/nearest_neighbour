import itertools
from itertools import count
from collections import Counter
import numpy as np
from utils import *



def NN_BF(L, R):
    assert len(L) == len(R)
    size = len(L)
    res = {}
    for i in range(size):
        for j in range(size):
            res[(i,j)] = np.sum(L[i] ^ R[j])
    return sorted(res.items(), key=lambda item: item[1])[0]

def distances(vectors, shift_vectors, iters):
    assert len(L) == len(R)
    m, size = len(L[0]), len(L)
    res = [[] for i in range(len(vectors))]
    for i in range(iters):
        shift_vec = shift_vectors[i]
        for j in range(size):
            dist = distance(vectors[j], shift_vec)
            res[j].append(dist)
    return res

def NN(L, R, iters = 1):
    assert len(L) == len(R)
    m, size = len(L[0]), len(L)
    shift_vectors =  [random_binary_vector(m) for _ in range(iters)]
    L1 = distances(L, shift_vectors, iters)
    R1 = distances(R, shift_vectors, iters)
    return L1, R1

# def create_mappings(dim):
#     """
#     for group size 1 - no change
#     for group size 2
#         [-2,-1]->-1 [ 0,1]->0, [2,3] -> 1 (operation: (distance-0) // 2)
#         [-3,-2]->-1 [-1,0]->0, [1,2] -> 1 (operation: (distance-1) // 2)
#     for group size 3
#         [-3,-2,-1]->-1 [0,1,2]  ->0, [3,4,5] -> 1 (operation: (distance-0) // (3))
#         [-4,-3,-2]->-1 [-1,0,1] ->0, [2,3,4] -> 1 (operation: (distance-1) // (3))
#         [-5,-4,-3]->-1 [-2,-1,0]->0, [1,2,3] -> 1 (operation: (distance-2) // (3))
#     for group size 4
#         [-4,-3,-2,-1]->-1 [0,1,2,3]   -> 0, [4,5,6,7] -> 1 (operation: (distance-0) // 4 )
#         [-5,-4,-3,-2]->-1 [-1,0,1,2]  -> 0, [3,4,5,6] -> 1 (operation: (distance-1) // 4 )
#         [-6,-5,-4,-3]->-1 [-2,-1,0,1] -> 0, [2,3,4,5] -> 1 (operation: (distance-2) // 4 )
#         [-7,-6,-5,-4]->-1 [-3,-2,-1,0]-> 0, [1,2,3,4] -> 1 (operation: (distance-3) // 4 )
#     """
#     res = {}
#     for group_size in range(0, dim):




def process_DV(dim, distances_vector, group_size):
    """
    for group size 1 - no change
    for group size 2
        [-2,-1]->-1 [ 0,1]->0, [2,3] -> 1 (operation: (distance-0) // 2)
        [-3,-2]->-1 [-1,0]->0, [1,2] -> 1 (operation: (distance-1) // 2)
    for group size 3
        [-3,-2,-1]->-1 [0,1,2]  ->0, [3,4,5] -> 1 (operation: (distance-0) // (3))
        [-4,-3,-2]->-1 [-1,0,1] ->0, [2,3,4] -> 1 (operation: (distance-1) // (3))
        [-5,-4,-3]->-1 [-2,-1,0]->0, [1,2,3] -> 1 (operation: (distance-2) // (3))
    for group size 4
        [-4,-3,-2,-1]->-1 [0,1,2,3]   -> 0, [4,5,6,7] -> 1 (operation: (distance-0) // 4 )
        [-5,-4,-3,-2]->-1 [-1,0,1,2]  -> 0, [3,4,5,6] -> 1 (operation: (distance-1) // 4 )
        [-6,-5,-4,-3]->-1 [-2,-1,0,1] -> 0, [2,3,4,5] -> 1 (operation: (distance-2) // 4 )
        [-7,-6,-5,-4]->-1 [-3,-2,-1,0]-> 0, [1,2,3,4] -> 1 (operation: (distance-3) // 4 )
    """


    res = [distances_vector]
    for position in range(dim):
        res = itertools.product(res, )

# def NN_Sys_basic(L, R, iters, shring_factor):
#     def shrink(val, shring_factor, center=0):
#         return int((val-center)/shring_factor)
#
#     assert len(L) == len(R)
#     m, size = len(L[0]), len(L)
#     center = m/2
#     L_1 = [[] for i in range(size)]
#     R_1 = [[] for i in range(size)]
#
#     for _ in range(iters):
#         for j in range(size):
#             shift = rand_vec(m)
#             lvec_dot_prod = np.sum(L[j] ^ shift)
#             rvec_dot_prod = np.sum(R[j] ^ shift)
#             l_prod = shrink(lvec_dot_prod, shring_factor, center=center)
#             r_prod = shrink(rvec_dot_prod, shring_factor, center=center)
#             L_1[j].append(l_prod)
#             R_1[j].append(r_prod)
#
#     return L_1, R_1





if __name__ == "__main__":
    L, R = gen_instance(m = 10, num_vectors = 5, NN_dist = 0,  pack = False)
    print(*L,'--'*40, *R, sep='\n')

    L1, R1 = NN(L, R, iters = 10)
    print(*L1,'--'*40, *R1, sep='\n')