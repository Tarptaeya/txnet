import numpy as np

class Sequential:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        n = len(self.layers)
        out = [0 for _ in range(n + 1)]
        out[0] = x
        for i in range(1, n + 1):
            out[i] = self.layers[i - 1].forward(out[i - 1])
        return out

    def fit(self, x, y, epochs=10):
        for _ in range(epochs):
            n = len(self.layers)
            out = self.forward(x)

            err = y - out[n]
            for i in range(n - 1, -1, -1):
                err = self.layers[i].backward(out[i], out[i + 1], err)

    def predict(self, x):
        return self.forward(x)[-1]

