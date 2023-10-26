import numpy as np
import seaborn as sns
import torch
from . import linearprog,sinkhorn
from .utils import cost_matrix, cost_matrix_1d, create_mask, subsequences
from ot.lp import emd_1d_sorted

def masking_map(xs, xt, lamb=5, s=None, sub_length=None, gamma=None, eps=1e-10, reg=0.0001, max_iterations=100000, thres=1e-5, algorithm="linear_programming", plot=False):
    '''
    Parameters
    ----------
        xs: ndarray, (m,d)
            d-dimensional source samples
        xt: ndarray, (n,d) 
            d-dimensional target samples
        lamb: lambda, int 
            Adjust the diagonal width. Default is 3
        algorithm: str
            algorithm to solve model. Default is "linear_programming". Choices should be
            "linear_programming" and "sinkhorn"
        plot: bool
            status for plot the optimal transport matrix or not. Default is "False"
    Returns
    ------- 
        cost: Transportation cost
    '''
    p = np.ones(len(xs))/len(xs)
    q = np.ones(len(xt))/len(xt)
    
    if xs.ndim == 1: 
        C = cost_matrix_1d(xs, xt)
    elif xs.ndim == 2:
        C = cost_matrix(xs, xt)
    else:
        raise ValueError("The data must in the form of 1d or 2d array")
    C /= (C.max() + eps)

    ## mask matrix
    M = create_mask(C, lamb)

    ## solving model
    if algorithm == "linear_programming":
        pi = linearprog.lp_partial(p,q,C,M)
    elif algorithm == "sinkhorn":
        pi = sinkhorn.sinkhorn_log_domain(p,q,C,M,reg,max_iterations,thres)
    else:
        raise ValueError("algorithm must be 'linear_programming' or 'sinkhorn'!")
    
    cost = np.sum(pi * C.numpy())
    #cost = np.exp(-gamma * cost)
    if plot:
        sns.heatmap(pi, linewidth=0.5)
        return pi, cost
    return cost

def masking_map_partial(xs, xt, lamb=5, s=0.5, sub_length=None, gamma=None, xi=None, A=None, eps=1e-10, reg=0.0001, max_iterations=100000, thres=1e-5, algorithm="linear_programming", plot=False):
    '''
    Parameters
    ----------
        xs: ndarray, (m,d)
            d-dimensional source samples
        xt: ndarray, (n,d) 
            d-dimensional target samples
        s: int
            The amount of mass wanted to transport through 2 empirical distribution
        lamb: lambda, int 
            Adjust the diagonal width. Default is 3
        data: 1d, 2d
            define the type of data
        algorithm: str
            algorithm to solve model. Default is "linear_programming". Choices should be
            "linear_programming" and "sinkhorn"
        plot: bool
            status for plot the optimal transport matrix or not. Default is "False"
    Returns
    ------- 
        cost: Transportation cost
    '''
    p = torch.Tensor(np.ones(len(xs))/len(xs))
    q = torch.Tensor(np.ones(len(xt))/len(xt))
    if xs.ndim == 1: 
        C = cost_matrix_1d(xs, xt)
    elif xs.ndim == 2:
        C = cost_matrix(xs, xt)
    else:
        raise ValueError("The data must in the form of 1d or 2d array")
    C /= (C.max() + eps)
    M = torch.Tensor(create_mask(C, lamb))

    #partial cost matrix
    if A is None:
        A = C.max()
    if xi is None:
        xi = 1e2*C.max()
    C_ = torch.cat((C, xi * torch.ones(C.size(0), 1)), dim=1)
    C_ = torch.cat((C_, xi * torch.ones(1, C_.size(1))), dim=0)
    C_[-1, -1] = 2 * xi + A

    # partial empirical distributions   
    p_ = torch.cat((p, (torch.sum(q) - s) * torch.Tensor([1])))
    q_ = torch.cat((q, (torch.sum(p) - s) * torch.Tensor([1])))

    # partial transportation mask
    a = torch.zeros(M.shape[0], 1, dtype=torch.int64)
    b = torch.zeros(M.shape[1] + 1, 1, dtype=torch.int64)
    M_ = torch.cat((M, a), dim=1)
    M_ = torch.cat((M_, b.t()), dim=0)
    pot = M_.shape[1]
    n, m = M_.shape
    for i in range(n-lamb*2, n):
        if (i > pot*n/m - lamb) & (i < pot*n/m + lamb):
            M_[i][pot-1]=1 
    
    pi_ = linearprog.lp_partial(p=p_.numpy(), q=q_.numpy(), C=C_.numpy(), Mask=M_.numpy())
    cost = np.sum(pi_ * C_.numpy())
    if plot:
        sns.heatmap(pi_, linewidth=0.5)
        return pi_, cost
    return cost

def masking_map_sequence(xs, xt, lamb=5, s=None, sub_length=25, gamma=0.1, eps=1e-10, reg=0.0001, max_iterations=100000, thres=1e-5, algorithm="linear_programming", cost_function="L2", plot=False):
    '''
    Parameters
    ----------
        a: ndarray, (m,d)
           d-dimensional source samples
        b: ndarray, (n,d) 
           d-dimensional target samples
        lamb: lambda, int 
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
    subs_xs = subsequences(xs, sub_length)
    subs_xt = subsequences(xt, sub_length)
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
    #cost = np.exp(-gamma * cost)
    if plot:
        sns.heatmap(pi, linewidth=0.5)
        return pi, cost
    return cost