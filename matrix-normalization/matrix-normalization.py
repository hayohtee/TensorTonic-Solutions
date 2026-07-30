import numpy as np


def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    A = np.array(matrix, dtype=float)

    if axis not in (None, 0, 1):
        return None
    
    if norm_type not in ("l2", "l1", "max"):
        return None

    if A.ndim != 2:
        return None

    norm = None

    match norm_type:
        case "l2":
            norm = np.sqrt(np.sum(A ** 2, axis=axis))
        case "l1":
            norm = np.sum(np.abs(A), axis=axis)
        case "max":
            norm = np.max(A, axis=axis)

    match axis:
        case 0:
            for i in range(A.shape[1]):
                norm_i = norm[i]
                A[:, i] = 0 if norm_i == 0 else A[:, i] / norm_i
        case 1:
            for i in range(A.shape[0]):
                norm_i = norm[i]
                A[i, :] = 0 if norm_i == 0 else A[i, :] / norm_i
        case None:
            A = 0 if norm == 0 else A / norm

    return A
