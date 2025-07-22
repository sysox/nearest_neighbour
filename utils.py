import sympy, random
import numpy as np
from math import log2
from scipy.optimize import root_scalar


################################################ Stats #################################################################

def binom_prob(n, HW, p=0.5):
    return round(sympy.binomial(n, HW) * p ** HW * (1 - p) ** (n - HW), 4)

def binom_dist(n):
    return [binom_prob(n, k) for k in range(0, n + 1)]

def cdf(dist):
    return [sum(dist[:i]) for i in range(1, len(dist) + 1)]

def center_cumulative(dist):
    return ([1] + [sum(dist[i:-i]) for i in range(1, len(dist)//2)])[::-1]

def distance_prob(distribution, distance):
    return sum([distribution[i]*distribution[i+distance] for i in range(len(distribution) - distance)])

def distance_allprobs(distribution):
    return [distance_prob(distribution, 0)] + [round(2*distance_prob(distribution, distance), 4) for distance in range(1, len(distribution))]


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

################################################ Numpy##################################################################

def random_binary_vector(m, pack = False, HW = None):
    bound = min(2**64, 2**m)
    num_uint64_chunks = (m + 63) // 64  # Calculate the number of uint64 chunks needed

    if pack == False:
        if HW == None:
            vec = np.random.randint(0, 2, m, dtype=np.uint8)
        else:
            vec = np.zeros(m, dtype=np.uint8)
            indices = np.random.choice(m, HW, replace=False)
            np.put(vec, indices, 1)
    else:
        if HW == None:
            vec = np.random.randint(low=0, high=min(2**64, 2**m), size=num_uint64_chunks, dtype=np.uint64)
        else:
            vec = random_binary_vector(m, pack = False, HW = HW)
            vec = np.packbits(vec, bitorder='little')
    return vec

def random_binary_vectors(m, **kwargs):
    """
    Generates two sets of 2**(lambda*m) random m-dimensional binary vectors.

    Args:
        m (int): The dimension of the vectors.
        lambda_val (float): A scalar for the size of the sets.

    Returns:
        tuple: Two NumPy arrays, each representing a set of binary vectors.
    """
    if 'lambda_val' in kwargs:
        num_vectors = 2 ** (kwargs['lambda_val'] * m)
    else:
        num_vectors = kwargs['num_vectors']
    num_vectors -= 1
    vectors = [random_binary_vector(m) for _ in range(int(num_vectors))]

    return vectors

def HW_int(n: np.uint64) -> int:
    return int(n).bit_count()

def HW_vector(vec) -> int:
    return sum([HW_int(val) for val in vec])


def gen_closest_vecs_instance(reference_vec, **kwargs):
    m = len(reference_vec)
    if 'gamma_val' in kwargs:
        HW = int(2 ** (kwargs['gamma_val'] * m))
    else:
        HW = kwargs['HW']
    indices_to_flip = np.random.choice(m, HW, replace=False)

if __name__ == '__main__':
    print(random_binary_vector(10, pack=False, HW=5))
    print(random_binary_vector(10, pack=True, HW=5))
