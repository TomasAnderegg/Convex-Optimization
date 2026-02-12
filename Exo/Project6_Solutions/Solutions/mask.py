import numpy as np
from scipy.special import comb

def mask(k, d):
    # Degree must be even
    if d % 2 != 0:
        print('ERROR: Degree must be even. Given degree is odd.')
        return None

    # Number of monomials in a polynomial of degree d and n = 2
    K = int(comb(2 + d, d))

    # Monomial index too high
    if k > K:
        print('ERROR: Monomial index too high.')
        return None

    # Matrix containing all possible index pairs (alpha_1, alpha_2)
    A = []
    for i in range(d + 1):
        for j in range(i + 1):
            A.append([i - j, j])
    A = np.array(A)

    # Size of quadratic form matrix for a polynomial of degree d and n = 2
    L = int(comb(2 + d // 2, d // 2))

    # Mask selecting those elements of the quadratic form matrix that add
    # up to the coefficient of the k-th monomial of the polynomial
    M = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            M[i, j] = np.sum(A[i, :] + A[j, :] == A[k - 1, :]) == 2  # Subtract 1 from k for 0-based indexing

    return M
