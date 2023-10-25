import numpy as np
import seaborn as sns
import torch
from . import utils,linearprog,sinkhorn,kpg_gw,partial_OT
from ot.lp import emd_1d_sorted

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

def cost_matrix_1d(x, y):
    x, y = torch.Tensor(x), torch.Tensor(y)
    Cxy = x.pow(2).sum(dim=1).unsqueeze(1) + y.pow(2).sum(dim=1).unsqueeze(0) - 2 * torch.matmul(x, y.t())
    #Cxy = np.expand_dims((x**2).sum(axis=1),1) + np.expand_dims((y**2).sum(axis=1),0) - 2 * x@y.T
    return Cxy

def create_mask(C, k):
    n, m = C.shape
    M = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if (i > j*n/m - k) & (i < j*n/m + k):
                M[i][j]=1 
    return M

def kpg_2d_rl_kp(xs, xt, gamma=0.1, lamb=3, sub_length=25, eps=1e-10, reg=0.0001, max_iterations=100000, thres=1e-5, algorithm="linear_programming", cost_function="L2", plot=False):
    p = np.ones(len(xs))/len(xs)
    q = np.ones(len(xt))/len(xt)

    C = cost_matrix(xs, xt)
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

def kpg_1d_distance(a, b, gamma=0.1, lamb=3, eps=1e-10, reg=0.0001, max_iterations=100000, thres=1e-5, algorithm="linear_programming", plot=False):
    '''
    Parameters
    ----------
        a: ndarray, (m,)
           d-dimensional source samples
        b: ndarray, (n,) 
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
    p = np.ones(len(a))/len(a)
    q = np.ones(len(b))/len(b)

    C = np.sum((a - b) ** 2, axis=1)
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

def partial_kpg_rl(xs, xt, gamma=0.1, lamb=3, sub_length=25, eps=1e-10, reg=0.0001, max_iterations=100000, thres=1e-5, algorithm="linear_programming", cost_function="L2", plot=False):
    p = np.ones(len(xs))/len(xs)
    q = np.ones(len(xt))/len(xt)

    C = cost_matrix(xs, xt)
    C /= (C.max() + eps)

    ## mask matrix
    M = create_mask(C, lamb)

    ## transport plan
    pi = partial_OT.partial_ot(torch.Tensor(p), torch.Tensor(q), torch.Tensor(G), I, J, s=s)
    return pi[:-1,:-1]

def partial_ot(p,q,C,I,J,s,xi=None,A=None,reg=0.001):
    if A is None:
        A = C.max()
    if xi is None:
        xi = 1e2*C.max()

    C_ = torch.cat((C, xi * torch.ones(C.size(0), 1)), dim=1)
    C_ = torch.cat((C_, xi * torch.ones(1, C_.size(1))), dim=0)
    C_[-1, -1] = 2 * xi + A

    M = torch.ones_like(C, dtype=torch.int64)
    M[I, :] = 0
    M[:, J] = 0
    M[I, J] = 1
    a = torch.ones(M.size(0), 1, dtype=torch.int64)
    a[I] = 0
    b = torch.ones(M.size(1) + 1, 1, dtype=torch.int64)
    b[J] = 0
    M_ = torch.cat((M, a), dim=1)
    M_ = torch.cat((M_, b.t()), dim=0)

    p_ = torch.cat((p, (torch.sum(q) - s) * torch.Tensor([1])))
    q_ = torch.cat((q, (torch.sum(p) - s) * torch.Tensor([1])))

    pi_ = linearprog.lp(p_.numpy(), q_.numpy(), C_.numpy(), M_.numpy())
    # pi_ = sinkhorn.sinkhorn_log_domain(p_, q_, C_, M_,reg=reg)
    # pi = pi_[:-1, :-1]
    return pi_