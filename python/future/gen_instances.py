import json, math, random
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


if __name__ == "__main__":
    num_vectors, match_prob, bit_size = 1000, 0.9, 64
    match_probs = [0.5+i/100 for i in range(10)] + [0.6, 0.7, 0.8] + [0.9+i/100 for i in range(10)]
    for num_vectors in [10, 100, 1000, 10000]:
        for match_prob in match_probs:
            to_save = [gen_instance(num_vectors=num_vectors, bit_size=bit_size, match_prob=match_prob) for _ in range(100)]
            json.dump(to_save, open(f"data/instances_{str(match_prob)[:4]}_{num_vectors}_{bit_size}.json", 'w'))