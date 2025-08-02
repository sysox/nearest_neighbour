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

def gen_instance(num_vectors, bit_size, match_prob):
    L = rand_integers(num_vectors, bit_size)
    R = rand_integers(num_vectors, bit_size)
    nn_dist = round((1 - match_prob) * bit_size)
    shift = rand_int_HW(HW=nn_dist, bit_size=bit_size)
    R[-1] = L[-1] ^ shift
    return L, R

################################ stats, complexity, params ################################


def dist_vectors(selected_L_pairs, selected_R_pairs, dist, func_operator = operator.le):
    for l in selected_L_pairs:
        for r in selected_R_pairs:
            if func_operator(HW(l ^ r), dist):
                return l, r
    return None, None

def NN(L, R, match_prob, bit_size=32, func_operator = operator.le):
    assert len(L) == len(R)
    num_vectors = len(L)
    dist = round((1-match_prob)*bit_size)
    l, r = dist_vectors(L, R, dist, func_operator)
    if (l, r) != (None, None):
        return {'success': (l, r) == (L[-1], R[-1]), "l": l, "r": r}

if __name__ == "__main__":
    num_vectors, match_prob, bit_size = 1000, 0.9, 64

    open('data/in')
    start = time.time()

    repetions = 1000
    for num_vectors in [10, 100, 1000, 10000]:
        success = []
        for _ in range(repetions):
            L, R = gen_instance(num_vectors=num_vectors, bit_size=bit_size, match_prob=match_prob)
            tmp = NN(L, R, match_prob, bit_size=bit_size)
            success.append(tmp['success'])
        print(f"time={time.time() - start}, success={sum(success)/repetions}, num_vectors={num_vectors}")

