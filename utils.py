import numpy as np
import scipy.sparse as sp
import torch
import pickle
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence


def new_load_data(src):
    print("load dataset...")
    with open(src + "/train_purposed.dat", 'rb') as f:
        data = pickle.load(f)
    train_X, train_y = data['X'], data['y']
    print("size: X (%s,%s,%s), y (%s,%s)" % tuple(train_X[0].shape + train_y.shape))

    with open(src + "/test_purposed.dat", 'rb') as f:
        data = pickle.load(f)
    test_X, test_y = data['X'], data['y']
    print("size: X (%s,%s,%s), y (%s,%s)" % tuple(test_X[0].shape + test_y.shape))

    train_X = [torch.Tensor(x) for x in train_X]
    train_y = torch.Tensor(train_y)
    test_X = [torch.Tensor(x) for x in test_X]
    test_y = torch.Tensor(test_y)

    with open("F:/DATA/dataset/v1/adj.dat", 'rb') as f:
        T_k = pickle.load(f)
    T_k = [sparse_mx_to_torch_sparse_tensor(m) for m in T_k]
    return train_X,  train_y, test_X, test_y, T_k


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        adj = adj.tocsr()
        d = d.tocsr()
        a_norm = adj.dot(d)
        a_norm = a_norm.transpose()
        a_norm = a_norm.dot(d)
        a_norm = a_norm.tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj


def normalized_laplacian(adj, symmetric=True):
    adj_normalized = normalize_adj(adj, symmetric)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    return laplacian


def rescale_laplacian(laplacian):
    try:
        print('Calculating largest eigenvalue of normalized graph Laplacian...')
        largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    except ArpackNoConvergence:
        print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2

    scaled_laplacian = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
    return scaled_laplacian


def chebyshev_polynomial(X, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = sp.csr_matrix(X, copy=True)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k+1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))
    T_k = [m.toarray() for m in T_k]
    T_k = [sp.csr_matrix(m, dtype = np.float32) for m in T_k]
    return T_k