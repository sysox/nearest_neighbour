import sympy, random
import numpy as np
from math import log2
from scipy.optimize import root_scalar


################################################ Entropy ###############################################################
def H(p):
    """
    Calculates the binary entropy function H(p).

    Args:
        p (float): The probability of success, a value between 0 and 1.

    Returns:
        float: The binary entropy in bits. Returns 0 if p is 0 or 1.
    """
    if p < 0 or p > 1:
        raise ValueError("Probability p must be between 0 and 1.")
    if p == 0 or p == 1:
        return 0
    return -p * log2(p) - (1 - p) * log2(1 - p)

def H_inv(target_entropy):
    """
    Calculates the inverse of the binary entropy function, specifically mapping
    the result 'p' to the interval [0, 0.5].

    Args:
        target_entropy (float): The entropy value (between 0 and 1).

    Returns:
        float: The probability 'p' corresponding to the target_entropy,
               where 0 <= p <= 0.5.

    Raises:
        ValueError: If target_entropy is out of valid range (0 to 1).
                    If a numerical root cannot be found within the [0, 0.5] interval.
    """
    if not (0 <= target_entropy <= 1):
        raise ValueError("Target entropy must be between 0 and 1.")

    # Handle edge cases explicitly where the result is known
    if target_entropy == 0:
        return 0.0
    if target_entropy == 1:
        return 0.5

    # Define the function whose root we want to find: H(p) - target_entropy = 0
    def f(p_val):
        return H(p_val) - target_entropy

    # Define the search interval for p in (0, 0.5).
    # Using small epsilon values to avoid log(0) and division by zero.
    lower_bound = 1e-10 # Very close to 0, but not 0
    upper_bound = 0.5 - 1e-10 # Very close to 0.5, but not 0.5

    try:
        # Use root_scalar with 'brentq' method for robustness within a bracket.
        # 'brentq' guarantees convergence if a root exists within the bracket
        # and the function changes sign across the bracket.
        sol = root_scalar(f, bracket=[lower_bound, upper_bound], method='brentq')

        if sol.converged:
            return sol.root
        else:
            # This case should be rare with 'brentq' if the bracket is chosen correctly
            # and a root exists, but it's good for explicit handling.
            raise ValueError(f"Numerical solver did not converge for target entropy {target_entropy}.")
    except Exception as e:
        # Catch potential errors from root_scalar itself (e.g., if no sign change in bracket)
        raise ValueError(f"Failed to find inverse for entropy {target_entropy} in [0, 0.5]. Error: {e}")

################################################ Numpy #################################################################
def binary_vector_to_uint64_array(binary_vector):
    vector_length = len(binary_vector)
    bits_needed_for_padding = (64 - (vector_length % 64)) % 64
    padded_vector_length = vector_length + bits_needed_for_padding

    packed_uint8 = np.packbits(binary_vector)

    # 4. Pad the uint8 array to be a multiple of 8 bytes for uint64 view
    bytes_needed_for_uint64_padding = (8 - (packed_uint8.size % 8)) % 8
    if bytes_needed_for_uint64_padding > 0:
        packed_uint8 = np.pad(packed_uint8, (0, bytes_needed_for_uint64_padding), 'constant', constant_values=0)

    uint64_array = packed_uint8.view(np.uint64)
    return uint64_array

def random_binary_vector(m, HW = None, pack = False):
    bound = min(2**64, 2**m)
    num_uint64_chunks = (m + 63) // 64  # Calculate the number of uint64 chunks needed

    if pack == False:
        if HW == None:
            vec = np.random.randint(0, 2, m, dtype=np.uint8)
        else:
            vec = np.zeros(m, dtype=np.uint8)
            assert m >= HW, print(m, HW)
            indices = np.random.choice(m, HW, replace=False)
            np.put(vec, indices, 1)
    else:
        if HW == None:
            vec = np.random.randint(low=0, high=min(2**64, 2**m), size=num_uint64_chunks, dtype=np.uint64)
        else:
            binary_vector = random_binary_vector(m, HW = HW, pack = False)
            vec = binary_vector_to_uint64_array(binary_vector)
    return vec

def random_binary_vectors(m, num_vectors = None, pack = False):
    vectors = [random_binary_vector(m, pack = pack) for _ in range(int(num_vectors))]
    return vectors

def gen_closest_vecs_instance(reference_vec, HW = 0, pack = False):
    m = len(reference_vec)
    vec_to_add = random_binary_vector(m, HW, pack)
    return reference_vec ^ vec_to_add

def gen_instance(m, num_vectors, NN_dist = 0,  pack = False):
    L = random_binary_vectors(m, num_vectors, pack)
    R = random_binary_vectors(m, num_vectors, pack)
    R[-1] = gen_closest_vecs_instance(L[-1], NN_dist, pack)
    return L, R

def HW_int(n: np.uint64) -> int:
    return int(n).bit_count()

def HW_vector(vec) -> int:
    return sum([HW_int(val) for val in vec])

def distance_positions(vec, shift_vec, indices):
    return sum([vec[indices[i]] ^ shift_vec[i] for i in range(len(indices))])

def distance(vec, shift_vec):
    return np.sum(vec ^ shift_vec)

def packed_vec_slice(packed_vec, idx_from, idx_to):
    start_block_idx = idx_from >> 6
    end_block_idx = (idx_to - 1) >> 6
    start_bit_offset = idx_from & 63
    num_bits_to_extract = idx_to - idx_from

    if start_block_idx == end_block_idx:
        element = packed_vec[start_block_idx]
        return [(element >> np.uint64(start_bit_offset)) & np.uint64(((1 << num_bits_to_extract) - 1))]
    else:
        lower_bits = packed_vec[start_block_idx] >> np.uint64(start_bit_offset)

        upper_bits_raw = packed_vec[end_block_idx]

        bits_from_first_elem = 64 - start_bit_offset
        bits_from_second_elem = num_bits_to_extract - bits_from_first_elem
        mask_for_upper = np.uint64((1 << bits_from_second_elem) - 1)
        upper_bits_shifted = (upper_bits_raw & mask_for_upper) << np.uint64(bits_from_first_elem)

        result = lower_bits | upper_bits_shifted
        assert 2**64 > result
        return [result]

def unpack_vector(vec, m = None):
    if m is None:
        m = len(vec) * 64
    full_binary_string = ''.join([format(chunk, '064b')[::-1] for chunk in vec])[:m]
    return list(map(int, full_binary_string))


if __name__ == '__main__':
    short = random_binary_vector(10, pack=False)
    short_HW = random_binary_vector(10, pack=False, HW=5)
    short_packed = random_binary_vector(10, pack=True)
    short_packed_HW = random_binary_vector(10, pack=True, HW=5)
    long_packed = random_binary_vector(96, pack=True)
    long_packed_HW = random_binary_vector(65, pack=True, HW=5)

    print(short)
    print(short_packed)
    print(unpack_vector(short_packed, m = 10))
    print(long_packed)
    assert HW_vector(short_HW) == 5
    assert HW_vector(short_packed_HW) == 5
    assert HW_vector(long_packed_HW) == 5

    print(unpack_vector(long_packed, m=96))
    print(unpack_vector(packed_vec_slice(long_packed, 0, 48)))
    print(unpack_vector(packed_vec_slice(long_packed, 48, 96)))

    print("Natural instance")
    NN_dist = random.randint(0, 2)
    L, R = gen_instance(m = 10, num_vectors = 4, NN_dist = NN_dist,  pack = False)
    print("L=", *L, '--' * 40, sep='\n')
    print( "R=", *R, '--' * 40, sep='\n')
    assert HW_vector(L[-1] ^ R[-1]) == NN_dist

    print("Packed instance")
    NN_dist = random.randint(0, 2)
    L, R = gen_instance(m = 10, num_vectors = 4, NN_dist = NN_dist,  pack = True)
    print("L=", *L, '--' * 40, sep='\n')
    print( "R=", *R, '--' * 40, sep='\n')
    assert HW_vector(L[-1] ^ R[-1]) == NN_dist