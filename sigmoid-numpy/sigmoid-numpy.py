import numpy as np

def sigmoid(x):
    if isinstance(x, float) or isinstance(x, int):
        return 1 / (1 + np.exp(-x))

    x = np.array(x)
    return 1 / (1 + np.exp(-x))