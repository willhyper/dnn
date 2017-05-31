'''
objective: two layer nn can learn XOR
'''
print(__doc__)
from pprint import pprint

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0, 1, 1, 0]).T  # XOR

model = Sequential()
model.add(Dense(4, input_dim=2, activation='sigmoid'))
model.add(Dense(1))
model.summary()

model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['accuracy'])

acc = 0
epochs = 0
while acc < 0.99:
    model.fit(X_train, y_train, epochs= 100, verbose=0)
    loss, acc = model.evaluate(X_train, y_train)
    epochs += 100
    print(epochs, loss, acc)

w = model.get_weights()
print('weights: ')
pprint(w)