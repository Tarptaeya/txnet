import numpy as np
from .activations import activ, deriv

class Dense:
    def __init__(self, n_input, n_output, activation='tanh', lr=0.001):
        self.n_input = n_input
        self.n_output = n_output
        self.weights = np.random.randn(n_input, n_output)
        self.bias = np.random.randn(1, n_output)
        self.activation = activation
        self.lr = lr

    def forward(self, x):
        out = np.dot(x, self.weights) + self.bias
        return activ(out, self.activation)

    def backward(self, inp, out, err):
        back_err = np.dot(err, self.weights.T)
        grad = err * deriv(out, self.activation)
        self.weights += self.lr * np.dot(inp.T, grad)
        self.bias += self.lr * np.dot(np.ones((1, grad.shape[0])), grad)
        return back_err


class Dropout:
    def __init__(self, p):
        self.p = p
        self.activ = None

    def forward(self, x):
        self.activ = np.random.binomial(1, self.p, size=x.shape)
        return x * self.activ

    def backward(self, inp, out, err):
        return err * self.activ

