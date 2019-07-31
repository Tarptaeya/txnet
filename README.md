# NNKit
Keras inspired pure python neural network library

```python3
from nnkit.models import Sequential
from nnkit.layers import Dense

import numpy as np
x = np.array([
  [1, 1],
  [0, 1],
  [1, 0],
  [0, 0]
])

y = np.array([[0], [1], [1], [0]])

model = Sequential()
model.add(Dense(2, 5))
model.add(Dense(5, 1))

model.fit(x, y)

print(model.predict(x))
```
