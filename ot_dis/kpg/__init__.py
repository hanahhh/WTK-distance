import numpy as np
import seaborn as sns

def subsequences(time_series, k):
    time_series = np.asarray(time_series)
    n = time_series.size
    shape = (n - k + 1, k)
    strides = time_series.strides * 2

    return np.lib.stride_tricks.as_strided(
        time_series,
        shape=shape,
        strides=strides
    )

def kpg_sequence_distance(a, b, kgot, l=3, sub_length=25, algorithm="linear_programming", plot=False):
    '''
    Parameters
    ----------
        a: ndarray, (m,d)
           d-dimensional source samples
        b: ndarray, (n,d) 
           d-dimensional target samples
        K: Class to access to the cost function
        l: lambda, int 
           Adjust the diagonal width. Default is 3
        sub_length: int
                    The number of elements of sub-sequence. Default is 25
        algorithm: str
                   algorithm to solve model. Default is "linear_programming". Choices should be
                   "linear_programming" and "sinkhorn"
        plot: bool
              status for plot the optimal transport matrix or not. Default is "False"
    Returns
    ------- 
        cost: Transportation cost
    '''
    subs_xs = subsequences(a, 25)
    subs_xt = subsequences(b, 25)
    p = np.ones(len(subs_xs))/len(subs_xs)
    q = np.ones(len(subs_xt))/len(subs_xt)
    pi, cost = kgot.kpg_rl_kp(p, q, subs_xs, subs_xt, K=l)
    if plot:
        sns.heatmap(pi, linewidth=0.5)
    return cost

def kpg_sequence_partial_distance(a, b, kgot, l=3, sub_length=25, algorithm="linear_programming", plot=False):
    '''
    Parameters
    ----------
        a: ndarray, (m,d)
           d-dimensional source samples
        b: ndarray, (n,d) 
           d-dimensional target samples
        K: Class to access to the cost function
        l: lambda, int 
           Adjust the diagonal width. Default is 3
        sub_length: int
                    The number of elements of sub-sequence. Default is 25
        algorithm: str
                   algorithm to solve model. Default is "linear_programming". Choices should be
                   "linear_programming" and "sinkhorn"
        plot: bool
              status for plot the optimal transport matrix or not. Default is "False"
    Returns
    ------- 
        cost: Transportation cost
    '''
    subs_xs = subsequences(a, 25)
    subs_xt = subsequences(b, 25)
    p = np.ones(len(subs_xs))/len(subs_xs)
    q = np.ones(len(subs_xt))/len(subs_xt)
    pi, cost = kgot.partial_kpg_rl(p, q, subs_xs, subs_xt, K=l)
    if plot:
        sns.heatmap(pi, linewidth=0.5)
    return cost