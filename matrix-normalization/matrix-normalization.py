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

    norms = None
    match norm_type:
        case "l2":
            norms = np.sqrt(np.sum(A ** 2, axis=axis, keepdims=True))
        case "l1":
            norms = np.sum(np.abs(A), axis=axis, keepdims=True)
        case "max":
            norms = np.max(A, axis=axis, keepdims=True)

    norms = np.where(norms == 0, 1, norms)

    return A / norms
