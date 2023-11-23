import torch
import numpy as np
import scipy.io as sio

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

def subsequence_2d(matrix_list, k):
    subsequences = [matrix_list[i:i + k] for i in range(0, len(matrix_list)-k)]
    return subsequences

def subsequence_2d_without_overlap(matrix_list, k):
    subsequences = [matrix_list[i:i + k] for i in range(0, len(matrix_list)-k, k)]
    return subsequences

def cost_matrix(x, y):
    x, y = torch.Tensor(x), torch.Tensor(y)
    Cxy = x.pow(2).sum(dim=1).unsqueeze(1) + y.pow(2).sum(dim=1).unsqueeze(0) - 2 * torch.matmul(x, y.t())
    #Cxy = np.expand_dims((x**2).sum(axis=1),1) + np.expand_dims((y**2).sum(axis=1),0) - 2 * x@y.T
    return Cxy

def cost_matrix_2d(x, y):
    m = len(x)
    n = len(y)
    Cxy = np.zeros((m, n))
    for row in range(m):
        for col in range(n):
            Cxy[row, col] = np.linalg.norm(x[row] - y[col])
    return Cxy

def cost_matrix_1d(x, y):
    x = torch.Tensor(x).view(-1, 1)
    y = torch.Tensor(y).view(1, -1)
    Cxy = (x - y) ** 2
    return Cxy

def create_mask(C, k):
    n, m = C.shape
    M = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if (i > j*n/m - k) & (i < j*n/m + k):
                M[i][j]=1 
    return M

def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x - np.max(x)))
    return f_x

def softmax_matrix(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x - np.max(x)), axis=-1, keepdims=True)
    return f_x


def KL_matrix(p, q, eps=1e-10):
    return np.sum(p * np.log(p + eps) - p * np.log(q + eps), axis=-1)


def JS_matrix(P, Q, eps=1e-10):
    P = np.expand_dims(P, axis=1)
    Q = np.expand_dims(Q, axis=0)
    kl1 = KL_matrix(P, (P + Q) / 2, eps)
    kl2 = KL_matrix(Q, (P + Q) / 2, eps)
    return 0.5 * (kl1 + kl2)

def create_neighbor_relationship(xs):
    xs = np.array(xs)
    if xs.ndim == 1:
        xt = np.insert(xs, 0, np.zeros_like(xs[0]))[:-1]
        f = xs - xt
        f = f.reshape(-1, 1)
    else:
        xt = np.vstack((np.zeros_like(xs[0]),xs))[:-1]
        f = xs - xt
    d = np.linalg.norm(f, axis=1)
    f1 = np.cumsum(d)
    sum_dist = f1[len(f1)-1]
    return f1/sum_dist

def create_mask_DT(xs, xt, lamb):
    f1 = create_neighbor_relationship(xs)
    f2 = create_neighbor_relationship(xt)
    n = len(f1)
    m = len(f2)
    M = np.zeros((n, m))
    for i in range(0, n):
        for j in range(0, m):
            if (i > j*n/m - lamb) & (i < j*n/m + lamb):
                if f1[i] == f2[j]:
                    M[i][j] = 1
                # elif np.abs(min(f1[i],f2[j])/max(f1[i],f2[j])) > lamb:
                #     M[i][j] = 1
                else:
                    M[i][j] = np.abs(min(f1[i],f2[j])/max(f1[i],f2[j]))
    return M