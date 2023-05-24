import numpy as np


def create_graph_PE(A, PE_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
        A: adjacency_matrix
        PE_dim: number of eigenvectors to use as PE
    """

    # Laplacian
    N = np.diag(np.power(A.sum(axis=1), -0.5))
    L = np.eye(A.shape[0]) - np.matmul(np.matmul(N, A), N)

    # Eigenvectors with numpy
    eig_val, eig_vec = np.linalg.eig(L)
    idx = eig_val.argsort() # increasing order
    eig_val, eig_vec = eig_val[idx], np.real(eig_vec[:, idx])
    pe = eig_vec[:,1:PE_dim + 1]

    # take the absolute value of pe
    # pe = np.abs(pe)
    
    # randomly flip the sign of each eigen vector
    # sign = -1 + 2 * np.random.randint(0, 2, (1, PE_dim))
    # pe *= sign
    
    return pe