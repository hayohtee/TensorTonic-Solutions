import numpy as np


def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """

    if p < 0 or p > 1:
        raise ValueError("Probability p must be in the range [0, 1].")
    
    x = np.asarray(x, dtype=float)
    pmf = np.empty_like(x)
    for i in range(len(pmf)):
        if x[i] == 1:
            pmf[i] = p
        elif x[i] == 0:
            pmf[i] = 1 - p
        else:
            raise ValueError("x must be 0 or 1 for Bernoulli distribution.")
    mean = p
    std = p * (1 - p)

    return pmf, mean, std