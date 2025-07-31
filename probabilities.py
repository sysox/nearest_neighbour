from utils import *
import prettytable

############################################ probabilities ############################################
def binom_prob(n, k, p=0.5, round_to=3):
    '''Calculate the binomial probability P(X = k) for a binomial distribution with parameters n and p.'''
    assert 0 <= p <= 1
    if k < 0 or k > n:
        return 0

    prob = sympy.binomial(n, k) * p ** k * (1 - p) ** (n - k)
    if round_to is not None:
        prob = round(prob, round_to)
    return prob

def pair_prob(n, k1, k2, match_prob, round_to=3):
    if k1 > k2:
        k1, k2 = k2, k1

    k1_prob = binom_prob(n, k1, p=0.5, round_to=round_to)

    # probability that for positions set to 1 in vector 1:
    # vector v2 matches v1 on l positions out of k1
    k2_prob = 0
    for match_k in range(k1+1):
        match_k_prob = binom_prob(n=k1, k=match_k, p=match_prob, round_to=None)
        match_on_rest = binom_prob(n=n-k1, k=k2 - match_k, p=1-match_prob, round_to=None)
        k2_prob += match_k_prob * match_on_rest
    res = k1_prob * k2_prob
    if round_to is not None:
        res = round(res, round_to)
    return res

############################################ Distribution ############################################

def binom_dist(n, p, round_to=3):
    return [binom_prob(n, k, p, round_to) for k in range(0, n + 1)]

def pair_dist(n, match_prob, round_to=3):
    res = []
    for i in range(n+1):
        probs = [pair_prob(n, i, j, match_prob, round_to) for j in range(n+1)]
        res.append(probs)
    return res

def pair_dist_table(n, match_prob, round_to=3):
    names = ['k1\k2'] + [str(i) for i in range(n+1)]
    table = prettytable.PrettyTable(names)
    s = 0
    rows = pair_dist(n, match_prob, round_to)
    sumation_row = ['sum'] + [round(sum(row), round_to) for row in rows] if round_to is not None else ['sum'] + [sum(row) for row in rows]
    # print(sum(sumation_row[1:]))
    for i in range(len(rows)):
        table.add_row([i] + rows[i])
    # assert s == 1, print(s)
    table.add_row(sumation_row)
    return table

def binomial_dist_table(n_max, p=0.5, round_to=3):
    names = ['n'] + [str(i) for i in range(n_max)]
    table = prettytable.PrettyTable(names)
    for n in range(1, n_max):
        probs = binom_dist(n, p, round_to)
        probs_padded = probs + [0] * (n_max - len(probs))
        table.add_row([n] + probs_padded)
    return table

def binomial_dist_table_half(n_max, p=0.5, round_to=3):
    size = n_max // 2
    names = ['n'] + [str(i) for i in range(size)]
    table = prettytable.PrettyTable(names)
    for n in range(1, n_max):
        probs = list(reversed(binom_dist(n, p, round_to)[:n//2 + 1]))
        probs_padded = probs + [0] * (size - len(probs))
        table.add_row([n] + probs_padded)
    return table

############################################ Cumulative ############################################
################### 1D #####################

def cdf(dist):
    return [sum(dist[:i]) for i in range(1, len(dist) + 1)]

def center_cumulative(dist):
    return ([1] + [sum(dist[i:-i]) for i in range(1, len(dist)//2)])[::-1]

def distance_prob(distribution, distance):
    return sum([distribution[i]*distribution[i+distance] for i in range(len(distribution) - distance)])

def distance_allprobs(distribution):
    return [distance_prob(distribution, 0)] + [round(2*distance_prob(distribution, distance), 4) for distance in range(1, len(distribution))]

################### 2D #####################
def match_probability(groups, pair_dist):
    res = 0
    for group in groups:
        for row in group:
            for col in group:
                res += pair_dist[row][col]
    return res

if __name__ == '__main__':
    # print(binomial_dist_table(10))

    print(pair_dist_table(n=1, match_prob=0.9, round_to=3))
    print(pair_dist_table(n=1, match_prob=0.5, round_to=3))
    for n in range(1, 10):
        diag = [[i] for i in range(n+1)]
        match_prob_nn = match_probability(groups=diag,pair_dist=pair_dist(n, match_prob=0.90, round_to=3))
        match_prob_rr = match_probability(groups=diag, pair_dist=pair_dist(n, match_prob=0.5, round_to=3))
        ratio = match_prob_nn / match_prob_rr
        print(f"n={n}, match_prob_nn={match_prob_nn:.2f}, match_prob_rr={match_prob_rr:.2f}, ratio={ratio:.2f}")

    print('--'*10)
    n = 3
    diag = [[0, 1], [2, 3]]
    match_prob_nn = match_probability(groups=diag,pair_dist=pair_dist(n, match_prob=0.9, round_to=3))
    match_prob_rr = match_probability(groups=diag, pair_dist=pair_dist(n, match_prob=0.5, round_to=3))
    ratio = match_prob_nn / match_prob_rr
    print(f"n={n}, match_prob_nn={match_prob_nn:.2f}, match_prob_rr={match_prob_rr:.2f}, ratio={ratio:.2f}")

    diag = [[0], [1, 2], [3]]
    match_prob_nn = match_probability(groups=diag, pair_dist=pair_dist(n, match_prob=0.9, round_to=3))
    match_prob_rr = match_probability(groups=diag, pair_dist=pair_dist(n, match_prob=0.5, round_to=3))
    ratio = match_prob_nn / match_prob_rr
    print(f"n={n}, match_prob_nn={match_prob_nn:.2f}, match_prob_rr={match_prob_rr:.2f}, ratio={ratio:.2f}")

    n = 11
    diag = [[0,1], [2,3], [4, 5, 6], [7,8], [9,10]]
    match_prob_nn = match_probability(groups=diag, pair_dist=pair_dist(n, match_prob=0.9, round_to=3))
    match_prob_rr = match_probability(groups=diag, pair_dist=pair_dist(n, match_prob=0.5, round_to=3))
    ratio = match_prob_nn / match_prob_rr
    print(f"n={n}, match_prob_nn={match_prob_nn:.2f}, match_prob_rr={match_prob_rr:.2f}, ratio={ratio:.2f}")



