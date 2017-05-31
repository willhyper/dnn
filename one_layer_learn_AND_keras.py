'''
objective: one layer nn can learn AND. No activation needed.
'''
print(__doc__)

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0, 0, 0, 1]).T  # AND

# Reshape your data so that each image becomes a long vector
N_train = X_train.shape[0]

model = Sequential()
model.add(Dense(1, input_dim=2))
model.summary()

# https://keras.io/optimizers/
model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['accuracy'])

epochs = 0
acc = 0
while acc < 0.99:
    model.fit(X_train, y_train, epochs=100, verbose=0)
    loss, acc = model.evaluate(X_train, y_train)
    epochs += 500
    print(epochs, loss, acc)

w = model.get_weights()
print('weights', w)
