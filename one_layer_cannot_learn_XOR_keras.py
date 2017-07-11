'''
objective: one layer nn cannot learn XOR.
note:
(1) no activation is needed for 1 layer nn as it is irrelevant
(2) metrics=['accuracy'] is required, otherwise TypeError: 'numpy.float64' object is not iterable
(3) (optimizer, loss) can be other combination

'''

print(__doc__)

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0, 1, 1, 0]).T  # XOR

model = Sequential()
model.add(Dense(1, input_dim=2))
model.summary()

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

while True:
    model.fit(X_train, y_train, epochs=1, verbose=0)
    loss, acc = model.evaluate(X_train, y_train, verbose=0)
    if acc >= 0.75: break

print('accuracy is up to 0.75. Math guaranteed.')
y_pred_class = model.predict_classes(X_train)[:, 0] # [0] = best guess
assert sum(y_pred_class == y_train) == 3