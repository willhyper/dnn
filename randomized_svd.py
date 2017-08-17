import numpy as np

def randomized_svd(M, k):
    m, n = M.shape
    transpose = False
    if m < n:
        transpose = True
        M = M.T

    rand_matrix = np.random.normal(size=(M.shape[1], k))  # short side by k
    Q, _ = np.linalg.qr(M @ rand_matrix, mode='reduced')  # long side by k
    smaller_matrix = Q.T @ M                              # k by short side
    U_hat, s, V = np.linalg.svd(smaller_matrix, full_matrices=False)
    U = Q @ U_hat

    if transpose:
        return V.T, s.T, U.T
    else:
        return U, s, V
