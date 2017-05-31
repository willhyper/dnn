'''
objective: one layer nn cannot learn XOR. The activation is irrelevant
'''
print(__doc__)

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0, 1, 1, 0]).T  # XOR

# Reshape your data so that each image becomes a long vector
N_train = X_train.shape[0]

model = Sequential()
model.add(Dense(1, input_dim=2))
model.summary()

# https://keras.io/optimizers/
model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['accuracy'])

for e in range(10):
    epoch = 100
    model.fit(X_train, y_train, epochs=epoch, verbose=0)
    loss, acc = model.evaluate(X_train, y_train)
    print(epoch * e, loss, acc)

print('well....accuracy will never be driven down. Math guaranteed.')
print('Stopped iterations.')
