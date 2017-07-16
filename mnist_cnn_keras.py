"""
objective: Learn MNIST to about 98% accuracy using CNN
feature simple model and in 3 epochs.
"""

"""
60000/60000 [==============================] - 18s - loss: 0.2408 - acc: 0.9301
Epoch 1/1
60000/60000 [==============================] - 18s - loss: 0.0846 - acc: 0.9734
Epoch 1/1
60000/60000 [==============================] - 19s - loss: 0.0615 - acc: 0.9812
Accuracy 0.9825 elapsed sec 57.116536140441895 n_param 37738
"""
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
import numpy as np
import time


class Codec:
    def enc(self, x): return to_categorical(x, 10)
    def dec(self, y): return np.argmax(y)
codec = Codec()

print(__doc__)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255.0
print('X_train.shape', X_train.shape)
print('X_test.shape', X_test.shape)

yc_train = codec.enc(y_train)
yc_test = codec.enc(y_test)
print('yc_train.shape', yc_train.shape)
print('yc_test.shape', yc_test.shape)

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

start = time.time()
while True:
    hist = model.fit(X_train, yc_train, epochs=1, verbose=1)
    acc_of_last_epoch = hist.history['acc'][-1]
    if acc_of_last_epoch > 0.98: break
elapse = time.time() - start
loss, acc = model.evaluate(X_test, yc_test, verbose=0)
n_param = model.count_params()
print('Accuracy', acc, 'elapsed sec', elapse, 'n_param', n_param)
