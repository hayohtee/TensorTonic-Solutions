import numpy as np
from numpy.linalg import det, inv

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    A = np.array(A)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return None

    if det(A) == 0:
        return None

    return inv(A)
