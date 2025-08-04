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

def masked_histogram(vectors: list[int], mask: int) -> dict[int, list[int]]:
    """
    Groups vectors into a dictionary based on their masked value.

    Args:
        vectors (list[int]): A list of integers to be grouped.
        mask (int): The bitmask to apply to each integer.

    Returns:
        dict[int, list[int]]: A dictionary where keys are the result of
                              (vector & mask) and values are lists of the
                              original vectors that produced that key.
    """
    # Use a defaultdict(list) to automatically handle the creation of
    # new lists for unseen keys. This is clean and efficient.
    histogram = defaultdict(list)
    for vec in vectors:
        key = vec & mask
        histogram[key].append(vec)
    return histogram

def NN_stats(L, R, mask):
    L_hashes = masked_histogram(L, mask)
    R_hashes = masked_histogram(R, mask)

    common_keys = set(L_hashes.keys()) & set(R_hashes.keys())
    complexity = 0
    for key in common_keys:
        complexity += len(L_hashes[key]) * len(R_hashes[key])

    if (L[-1] & mask) != (R[-1] & mask):
        complexity = -complexity
    return complexity

def stats(size = 100, match_prob = 0.9, max_subset=20, repetitions = 10**3):
    L, R = gen_instance(size, 32, match_prob)
    res = {}
    for mask_HW in range(max_subset):
        complexities = []
        for _ in range(repetitions):
            random_mask = rand_int_HW(bit_size=32, HW=mask_HW)
            complexities.append(NN_stats(L, R, random_mask))
        success_rate = sum([1 for complexity in complexities if complexity > 0]) / len(complexities)
        average_complexity = sum(map(abs, complexities)) / len(complexities)
        res[mask_HW] = {'success_rate': success_rate, 'BF_complexity': average_complexity}
    return res

def compute_stats(file_name, repetitions):
    smallest_probs = [i/100 for i in range(50, 60)]
    highest_probs = [i / 100 for i in range(91, 101)]
    middle_probs = [i / 100 for i in range(60, 91, 5)]
    match_probs = smallest_probs + middle_probs + highest_probs
    res = {}
    for match_prob in match_probs:
        res[match_prob] = stats(match_prob=match_prob, repetitions=repetitions)
        print(match_prob)
    with open(file_name, 'w') as f:
        json.dump(res, f, indent=4)

def complexity(num_vectors, match_prob, mask_HW):
    Complexity = namedtuple("Complexity", ["complexity", "num_repetitions", "BFcomplexity", 'mask_HW'])
    num_repetitions = 1 / pow(match_prob, mask_HW)
    BFcomplexity = pow(num_vectors, 2) / pow(2, mask_HW)
    BFcomplexity = max(BFcomplexity, 1) # complexity is at least 1
    return Complexity(complexity=num_repetitions*BFcomplexity, num_repetitions=num_repetitions, BFcomplexity=BFcomplexity, mask_HW=mask_HW)

def find_best_complexity(num_vectors, match_prob):
    sort_complexity = num_vectors*math.log2(num_vectors)
    complexities = [complexity(num_vectors, match_prob, i) for i in range(32)]
    filtered_complexities = [item for item in complexities if item.complexity > sort_complexity]
    filtered_sorted = sorted(filtered_complexities, key=lambda x: x.complexity)
    return filtered_sorted[0]

################################ Algorithm ################################
'''
Based on sort:
'''
def cross_match_idxs(L_pairs_sorted, R_pairs_sorted, max_idx, idx_l_from=0, idx_r_from=0):
    idx_l, idx_r = idx_l_from, idx_r_from
    while idx_l < max_idx and idx_r < max_idx:
        if L_pairs_sorted[idx_l].masked == R_pairs_sorted[idx_r].masked:
            break
        if L_pairs_sorted[idx_l].masked < R_pairs_sorted[idx_r].masked:
            idx_l += 1
        else:
            idx_r += 1
    return idx_l, idx_r

def same_match_idxs(X_pair_sorted, max_idx, idx_from):
    idx = idx_from
    while (idx + 1 < max_idx) and (X_pair_sorted[idx].masked == X_pair_sorted[idx+1].masked):
        idx += 1
    return idx_from, idx + 1

def dist_vectors(selected_L_pairs, selected_R_pairs, dist, func_operator = operator.le):
    for l_pair in selected_L_pairs:
        for r_pair in selected_R_pairs:
            if func_operator(HW(l_pair.vec ^ r_pair.vec), dist):
                return l_pair.vec, r_pair.vec
    return None, None

def NN(L, R, match_prob, mask_HW=None, bit_size=32, func_operator = operator.le):
    Vec_pair = namedtuple("Vec_pair", ["vec", "masked"])

    assert len(L) == len(R)
    num_vectors = len(L)
    dist = round((1-match_prob)*bit_size)
    if mask_HW is None:
        mask_HW = find_best_complexity(num_vectors, match_prob).mask_HW

    repetitions = 1
    vector_comparisons = 0
    while True:
        mask = rand_int_HW(HW=mask_HW, bit_size=bit_size)
        L_pairs = [Vec_pair(vec=l, masked=l & mask) for l in L]
        R_pairs = [Vec_pair(vec=r, masked=r & mask) for r in R]
        L_pairs_sorted = sorted(L_pairs, key=lambda vec_pair: vec_pair.masked)
        R_pairs_sorted = sorted(R_pairs, key=lambda vec_pair: vec_pair.masked)

        idx_l, idx_r = 0, 0
        while (idx_l < num_vectors) and (idx_r < num_vectors):
            idx_l, idx_r = cross_match_idxs(L_pairs_sorted, R_pairs_sorted, num_vectors, idx_l, idx_r)
            if idx_l == num_vectors or idx_r == num_vectors:
                break
            idx_from, idx_to = same_match_idxs(L_pairs_sorted, num_vectors, idx_l)
            matching_masked_L_pairs = L_pairs_sorted[idx_from:idx_to]
            idx_l = idx_to
            idx_from, idx_to = same_match_idxs(R_pairs_sorted, num_vectors, idx_r)
            matching_masked_R_pairs = R_pairs_sorted[idx_from:idx_to]
            idx_r = idx_to

            vector_comparisons += len(matching_masked_L_pairs)*len(matching_masked_R_pairs)

            l, r = dist_vectors(matching_masked_L_pairs, matching_masked_R_pairs, dist, func_operator)
            if (l, r) != (None, None):
                return {'success': (l, r) == (L[-1], R[-1]), "l": l, "r": r, "repetitions": repetitions, "vector_comparisons":vector_comparisons, 'mask_HW':mask_HW}
        repetitions += 1

if __name__ == "__main__":
    num_vectors, match_prob, bit_size = 10000, 0.9, 64

    for mask_HW in range(6, 15):
        start = time.time()
        success, repetitions, vector_comparisons = [], [], []
        for _ in range(100):
            L, R = gen_instance(num_vectors=num_vectors, bit_size=bit_size, match_prob=match_prob)
            tmp = NN(L, R, match_prob, bit_size=bit_size, mask_HW=mask_HW)
            success.append(tmp['success'])
            repetitions.append(tmp['repetitions'])
            vector_comparisons.append(tmp['vector_comparisons'])
        print(f" mask_HW={mask_HW} time={time.time() - start}, #success={sum(success)}, #repetitions={sum(repetitions)/10**6}, #vector_comparisons={sum(vector_comparisons)/10**6}")
    # print(res)
    # l, r = res['l'], res['r']
    # print(dist(l , r))
    # print(binary(l, bit_size))
    # print(binary(r, bit_size))
