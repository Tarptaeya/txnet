import numpy as np

__activations = {
    'tanh': [lambda x: np.tanh(x), lambda x: 1 - x * x],
    'sigmoid': [lambda x: 1 / (1 + np.exp(-x)), lambda x: x * (1 - x)],
    'relu': [lambda x: np.maximum(x, 0), lambda x: np.where(x <= 0, 0, 1)],
}

def activ(out, fn):
    return __activations[fn][0](out)

def deriv(out, fn):
    return __activations[fn][1](out)

