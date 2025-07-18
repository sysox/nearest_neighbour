import sympy, random
import numpy as np
from math import log2
import prettytable
from scipy.optimize import root_scalar

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



def rand_vec(m):
    return np.random.randint(0, 2, size=m, dtype=bool)

def generate_random_binary_sets(m, **kwargs):
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

    # Use dtype=bool for memory efficiency if values are strictly 0 or 1
    # Otherwise, use np.uint8 for bitwise operations if needed later
    set1 = np.random.randint(0, 2, size=(int(num_vectors), m), dtype=bool)
    set2 = np.random.randint(0, 2, size=(int(num_vectors), m), dtype=bool)
    return set1, set2


def generate_vector_with_hamming_distance(reference_vector, **kwargs):
    """
    Generates a random binary vector with a specific Hamming distance from a reference vector.

    Args:
        reference_vector (np.ndarray): The base binary vector (1D NumPy array).
        gamma_val (float): A scalar for the target Hamming distance.

    Returns:
        np.ndarray: A new binary vector with the specified Hamming distance.
    """
    m = len(reference_vector)
    if 'gamma_val' in kwargs:
        HW = int(2 ** (kwargs['gamma_val'] * m))
    else:
        HW = kwargs['HW']
    if HW > m:
        raise ValueError(f"Target Hamming distance ({HW}) cannot exceed vector dimension ({m}).")

    # Get indices to flip
    # Randomly choose 'target_distance' unique indices to flip
    indices_to_flip = np.random.choice(m, HW, replace=False)

    # Create a copy to modify
    new_vector = reference_vector.copy()

    # Flip the bits at selected indices
    # For boolean arrays, XORing with True (or 1) flips the bit
    # For integer arrays, XORing with 1 flips the bit
    new_vector[indices_to_flip] = ~new_vector[indices_to_flip]  # For boolean
    # If new_vector was int: new_vector[indices_to_flip] = 1 - new_vector[indices_to_flip] or new_vector[indices_to_flip] ^= 1

    return new_vector


def gen_closest_vecs_instance(m, **kwargs):
    """
    Generates two lists of vectors, L and R, and a special pair (l, r)
    such that l is appended to L, r is appended to R, and the Hamming
    distance between l and r is controlled by the HW/gamma_val parameter.

    Args:
        m (int): The dimension of the vectors.
        **kwargs: Keyword arguments passed to the generation functions.

    Returns:
        tuple: The updated L and R NumPy arrays, each with one additional vector.
    """
    L, R = generate_random_binary_sets(m, **kwargs)

    # Generate the special "closest pair" vectors
    l = np.random.randint(0, 2, size=m, dtype=bool)
    r = generate_vector_with_hamming_distance(l, **kwargs)

    # Use np.vstack to efficiently append the new vectors to their respective sets.
    # This stacks the arrays vertically, creating new arrays with one additional row.
    L_with_l = np.vstack((L, l))
    R_with_r = np.vstack((R, r))

    return L_with_l, R_with_r

def np2clasical_list(vectors):
    return list(map(list, vectors.astype(int)))

if __name__ == '__main__':
    L, R = gen_closest_vecs_instance(m=10, num_vectors=3, HW=2)
    print(L.astype(int))
    print(R.astype(int))
    print((L[-1] ^ R[-1]).astype(int))



    # y_values = [[j/10] + [ y_comp(gamma_val=0.5*i/10, lambda_val=j/10) for i in range(11)] for j in range(11)]
    # T = prettytable.PrettyTable()
    # T.add_rows(y_values)
    # print(T)


