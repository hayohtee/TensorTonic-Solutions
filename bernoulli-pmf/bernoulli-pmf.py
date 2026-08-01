import numpy as np


def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """

    if p < 0 or p > 1:
        raise ValueError("Probability p must be in the range [0, 1].")
    
    x = np.asarray(x, dtype=float)
    pmf = np.where(x == 1, p, 1-p)
    mean = p
    std = p * (1 - p)

    return pmf, mean, std