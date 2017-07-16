"""
objective: one layer NN can learn MNIST to about 92% accuracy
using softmax as activation so that output represents probability
loss = 'categorical_crossentropy ' is required by to_categorical.
.astype('float32')/ 255.0 is critical for improving accuracy

"""
print(__doc__)
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np

class Codec:
    def enc(self, x): return to_categorical(x, 10)
    def dec(self, y): return np.argmax(y)
codec = Codec()

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

X_train = X_train.reshape(X_train.shape[0], 28 * 28).astype('float32') / 255.0
X_test = X_test.reshape(X_test.shape[0], 28 * 28).astype('float32') / 255.0
yc_train = codec.enc(y_train)
yc_test = codec.enc(y_test)
print('yc_train.shape', yc_train.shape)
print('yc_test.shape', yc_test.shape)

model = Sequential()
model.add(Dense(10, input_dim=784, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

while True:
    model.fit(X_train, yc_train, epochs=1, verbose=0)
    loss, acc = model.evaluate(X_test, yc_test)
    if acc > 0.92: break

print('Accuracy %f is achieved' % acc)
def view_sample():
    ind = np.random.randint(0, yc_test.shape[0])
    Xind = X_test[np.array([ind])]
    y_predict = codec.dec(model.predict(Xind)) # model.predict_classes(Xind)[0]
    y_true = codec.dec(yc_test[ind])
    print('y_true is ', y_true, '. Machine predicts', y_predict)

for _ in range(10):
    view_sample()
