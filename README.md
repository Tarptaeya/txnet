# txnet
Keras inspired pure python neural network library

```python3
from txnet.models import Sequential
from txnet.layers import Dense, ReLU, Sigmoid

import numpy as np
x = np.array([
  [1, 1],
  [0, 1],
  [1, 0],
  [0, 0]
])

y = np.array([[0], [1], [1], [0]])

model = Sequential()
model.add(Dense(2, 5, 0.01))
model.add(ReLU())
model.add(Dense(5, 1, 0.01))
model.add(Sigmoid())

model.fit(x, y, 10000)

print(model.predict(x))
```
