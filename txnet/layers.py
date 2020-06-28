import numpy as np


class AbstractLayer:
    def forward(self, X):
        raise NotImplementedError()

    def backward(self, derr_dout):
        raise NotImplementedError()


class Sigmoid(AbstractLayer):
    def forward(self, X):
        self.X = X
        return self.sig(X)

    def backward(self, derr_dout):
        return derr_dout * self.sig(self.X) * self.sig(1 - self.X)

    def sig(self, v):
        return 1 / (1 + np.exp(-v))


class ReLU(AbstractLayer):
    def forward(self, X):
        self.X = X
        return X * (X > 0)

    def backward(self, derr_dout):
        return 1.0 * (self.X > 0) * derr_dout


class Tanh(AbstractLayer):
    def forward(self, X):
        self.X = X
        return np.tanh(X)

    def backward(self, derr_dout):
        return (1 - self.X * self.X) * derr_dout


class Dense(AbstractLayer):
    def __init__(self, n_in, n_out, lr=0.01):
        self.n_in = n_in
        self.n_out = n_out
        self.lr = lr
        self.w = np.random.randn(n_in, n_out)
        self.b = np.random.randn(1, n_out)

    def forward(self, X):
        self.X = X
        self.out = X @ self.w + self.b
        return self.out

    def backward(self, derr_dout):
        derr_dw = self.X.T @ derr_dout
        derr_db = derr_dout
        derr_din = derr_dout @ self.w.T

        self.w = self.w - self.lr * derr_dw
        self.b = self.b - self.lr * derr_db

        return derr_din

class Droupout(AbstractLayer):
    def __init__(self, p):
        self.p = p

    def forward(self, X):
        self.X = np.random.binomial(1, self.p, size=X.shape)
        return X * self.X

    def backward(self, derr_dout):
        return derr_dout * self.X
