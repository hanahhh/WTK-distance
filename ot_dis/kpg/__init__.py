import numpy as np
import seaborn as sns
import torch
from . import utils,linearprog,sinkhorn,kpg_gw,partial_OT

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

def cost_matrix(x, y):
    x, y = torch.Tensor(x), torch.Tensor(y)
    Cxy = x.pow(2).sum(dim=1).unsqueeze(1) + y.pow(2).sum(dim=1).unsqueeze(0) - 2 * torch.matmul(x, y.t())
    #Cxy = np.expand_dims((x**2).sum(axis=1),1) + np.expand_dims((y**2).sum(axis=1),0) - 2 * x@y.T
    return Cxy

def create_mask(C, k):
    M = np.zeros_like(C)
    n, m = M.shape
    for i in range(n):
        for j in range(m):
            if (i > j*n/m - k) & (i < j*n/m + k):
                M[i][j]=1 
    return M

def kpg_sequence_distance(a, b, gamma=0.1, lamb=3, sub_length=25, eps=1e-10, reg=0.0001, max_iterations=100000, thres=1e-5, algorithm="linear_programming", cost_function="L2", plot=False):
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
    subs_xs = subsequences(a, sub_length)
    subs_xt = subsequences(b, sub_length)
    p = np.ones(len(subs_xs))/len(subs_xs)
    q = np.ones(len(subs_xt))/len(subs_xt)

    C = cost_matrix(subs_xs, subs_xt)
    C /= (C.max() + eps)

    ## mask matrix
    M = create_mask(C, lamb)

    ## solving model
    if algorithm == "linear_programming":
        pi = linearprog.lp(p,q,C,M)
    elif algorithm == "sinkhorn":
        pi = sinkhorn.sinkhorn_log_domain(p,q,C,M,reg,max_iterations,thres)
    else:
        raise ValueError("algorithm must be 'linear_programming' or 'sinkhorn'!")
    D = C.numpy()
    cost = np.sum(pi * D)
    cost = np.exp(-gamma * cost)
    if plot:
        sns.heatmap(pi, linewidth=0.5)
        return pi, cost
    return cost

def kpg_sequence_partial_distance(a, b, lamb=3, s=0.5, sub_length=25, eps=1e-10, reg=0.0001, max_iterations=100000, thres=1e-5, algorithm="linear_programming", cost_function="L2", plot=False):
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

    C = cost_matrix(subs_xs, subs_xt)
    C /= (C.max() + eps)

    ## mask matrix
    M = create_mask(C, lamb)
    
    ## transport plan
    pi = partial_OT.partial_ot(torch.Tensor(p), torch.Tensor(q), torch.Tensor(C), s=s, M=M)
    D = C.numpy()
    cost = np.sum(pi * D)

    if plot:
        sns.heatmap(pi, linewidth=0.5)
        return pi[:-1,:-1], cost
    return cost

    # C = cost_matrix(subs_xs, subs_xt, cost_function,eps)
    # C /= (C.max() + eps)

    # ## mask matrix
    # M = create_mask(C, lamb)

    # ## solving model
    # if algorithm == "linear_programming":
    #     pi = linearprog.lp(p,q,C,M)
    # elif algorithm == "sinkhorn":
    #     pi = sinkhorn.sinkhorn_log_domain(p,q,C,M,reg,max_iterations,thres)
    # else:
    #     raise ValueError("algorithm must be 'linear_programming' or 'sinkhorn'!")
    # D = C.numpy()
    # cost = np.sum(pi * D)
    # if plot:
    #     sns.heatmap(pi, linewidth=0.5)
    # return pi, cost