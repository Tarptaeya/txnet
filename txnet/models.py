import numpy as np
import matplotlib.pyplot as plt

class Sequential:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def fit(self, X, y, n_epochs=10):
        for epoch in range(n_epochs):
            N = len(X)
            for i in range(N):
                out = X[i].reshape(1, -1)
                for l in self.layers:
                    out = l.forward(out)

                t = y[i].reshape(1, -1)
                derr_dout = -2 * (t - out).reshape(1, -1)

                for l in self.layers[::-1]:
                    derr_dout = l.backward(derr_dout)

    def predict(self, X):
        out = X
        for l in self.layers:
            out = l.forward(out)
        return out
