import math
import numpy as np
def cumulative_sum(a, i, s):
    k = math.floor(len(a)*s)
    if i>=k:
        return sum(a[:i]) - sum(a[:i-k])
    else:
        return sum(a[:i])
def otw_distance(a, b, m, s=1/5):
    sum_a = sum(a[:len(a)])
    sum_b = sum(b[:len(b)])
    a_hat = np.append(a, sum_b)
    b_hat = np.append(b, sum_a)
    n = len(a_hat)
    first_element = m * abs(cumulative_sum(a, n, s) - cumulative_sum(b, n, s))
    second_element = 0
    for i in range(1, n):
        second_element += abs(cumulative_sum(a, i, s) - cumulative_sum(b, i, s))
    return first_element + second_element