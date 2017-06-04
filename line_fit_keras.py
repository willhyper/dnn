'''
objective: fit y=x. expect weights = [slope intercept] = [1 0]
'''
from pprint import pprint
print(__doc__)

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_train = np.array([0, 1, 3, 4])
y_train = x_train


model = Sequential()
model.add(Dense(1, input_dim=1))
model.summary()

# https://keras.io/optimizers/
model.compile(optimizer='sgd',
              loss='mean_squared_error',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1000, verbose=0)
loss, acc = model.evaluate(x_train, y_train)
print(loss, acc)

w = model.get_weights()
print('weights')
pprint(w)
