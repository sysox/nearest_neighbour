from utils import *


print("probability distributions:")
for n in range(1, 20, 2):
    dist = binom_dist(n)
    half_idx = n//2+1
    print(n, dist[half_idx:])

print("cumulative distribution from the center:")
for n in range(1, 20, 2):
    dist = binom_dist(n)
    half_idx = n//2+1
    print(n, center_cumulative(dist))


print("probability distribution for the differences 0, 1, ... n:")
for n in range(1, 33, 2):
    dist = binom_dist(n)
    print(n, distance_allprobs(dist))