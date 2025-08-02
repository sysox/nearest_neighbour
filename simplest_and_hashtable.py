from collections import defaultdict, namedtuple
import json, math, random
import operator
import time
####################################### utils #######################################

def binary(vec_as_int, size=32):
    return format(vec_as_int, f'0{size}b')

def HW(v):
    return int(v).bit_count()

def dist(u, v):
    return HW(u^v)

####################################### generation #######################################

def rand_integers(num_vectors: int, bit_size: int) -> list[int]:
    upper_bound = 2**bit_size
    return [random.randrange(upper_bound) for _ in range(num_vectors)]

def rand_int_HW(bit_size: int, HW: int) -> int:
    if HW == 0:
        return 0
    one_indices = random.sample(range(bit_size), HW)
    result_int = sum(2**i for i in one_indices)
    return result_int

def rand_maskvalues_HW(bit_size: int, HW: int) -> int:
    if HW == 0:
        return 0
    one_indices = random.sample(range(bit_size), HW)
    return [2**i for i in one_indices]

def gen_instance(num_vectors, bit_size, match_prob):
    L = rand_integers(num_vectors, bit_size)
    R = rand_integers(num_vectors, bit_size)
    nn_dist = round((1 - match_prob) * bit_size)
    shift = rand_int_HW(HW=nn_dist, bit_size=bit_size)
    R[-1] = L[-1] ^ shift
    return L, R

################################ stats, complexity, params ################################

################################ Algorithm ################################
'''
Based on hash table:
'''

def dist_vectors(selected_L_pairs, selected_R_pairs, dist, func_operator = operator.le):
    for l_pair in selected_L_pairs:
        for r_pair in selected_R_pairs:
            if func_operator(HW(l_pair ^ r_pair), dist):
                return l_pair, r_pair
    return None, None
def bits_extraction(value, mask_values):
    res = 0
    for mask_value in mask_values:
        res = (res << 1)
        res |= (value & mask_value) == mask_value
    return res

def NN(L, R, match_prob, mask_HW=None, bit_size=32, func_operator = operator.le):
    Vec_pair = namedtuple("Vec_pair", ["vec", "masked"])
    assert len(L) == len(R)
    num_vectors = len(L)
    dist = round((1-match_prob)*bit_size)
    # if mask_HW is None:
    #     mask_HW = find_best_complexity(num_vectors, match_prob).mask_HW

    repetitions = 1
    vector_comparisons = 0
    table_size = 1 << mask_HW
    while True:
        L_table = [[] for _ in range(1<<mask_HW)]
        R_table = [[] for _ in range(1<<mask_HW)]
        mask_values = rand_maskvalues_HW(HW=mask_HW, bit_size=bit_size)
        for i in range(num_vectors):
            idx = bits_extraction(L[i], mask_values)
            L_table[idx].append(L[i])
            idx = bits_extraction(R[i], mask_values)
            R_table[idx].append(R[i])

        for i in range(table_size):
            l, r = dist_vectors(L_table[i], R_table[i], dist, func_operator)
            vector_comparisons += len(L_table[i])*len(R_table[i])
            if (l, r) != (None, None):
                return {'success': (l, r) == (L[-1], R[-1]), "l": l, "r": r, "repetitions": repetitions, "vector_comparisons":vector_comparisons, 'mask_HW':mask_HW}
            repetitions += 1

if __name__ == "__main__":
    num_vectors, match_prob, bit_size = 1000, 0.9, 64
    for mask_HW in range(7, 13):
        start = time.time()
        success, repetitions, vector_comparisons = [], [], []
        for _ in range(100):
            L, R = gen_instance(num_vectors=num_vectors, bit_size=bit_size, match_prob=match_prob)
            tmp = NN(L, R, match_prob, bit_size=bit_size, mask_HW=mask_HW)
            success.append(tmp['success'])
            repetitions.append(tmp['repetitions'])
            vector_comparisons.append(tmp['vector_comparisons'])
        print(f" mask_HW={mask_HW} time={time.time() - start}, #success={sum(success)}, #repetitions={sum(repetitions)}, #vector_comparisons={sum(vector_comparisons)}")

