import numpy as np
import matplotlib.pyplot as plt

class Sequential:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def fit(self, X, y, n_epochs=10):
        errors = []
        for epoch in range(n_epochs):
            N = len(X)
            for i in range(N):
                out = X[i].reshape(1, -1)
                for l in self.layers:
                    out = l.forward(out)

                t = y[i].reshape(1, -1)
                derr_dout = -2 * (t - out).reshape(1, -1)

                if epoch % (n_epochs // 20) == 0: errors.append(np.sum((t - out) ** 2))

                for l in self.layers[::-1]:
                    derr_dout = l.backward(derr_dout)

        plt.plot([i for i in range(len(errors))], errors)
        plt.show(block=True)

    def predict(self, X):
        out = X
        for l in self.layers:
            out = l.forward(out)
        return out
