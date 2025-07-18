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

if __name__ == "__main__":
    pass