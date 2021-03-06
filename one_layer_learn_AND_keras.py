'''
objective: one layer nn can learn AND.
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
y_train = np.array([[0], [0], [0], [1]])  # AND

model = Sequential()
model.add(Dense(1, input_dim=2))
model.summary()

model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])

while True:
    hist = model.fit(X_train, y_train, epochs=1, verbose=0)
    acc_of_last_epoch = hist.history['acc'][-1]
    if acc_of_last_epoch == 1.0: break

y_pred_class = model.predict_classes(X_train)
assert all(y_pred_class == y_train)
print('learned AND.')
