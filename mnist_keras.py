from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np

class codec:
    @staticmethod
    def enc(x):
        return to_categorical(x, 10)
    @staticmethod
    def dec(y):
        return np.argmax(y)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Data type:", X_train.dtype)

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

model.fit(X_train, yc_train, validation_split=0.3, epochs=1)

loss, acc = model.evaluate(X_test, yc_test, verbose=0)
print('evaluate. loss, acc = ', loss, acc)

def view_sample():
    ind = np.random.randint(0, yc_test.shape[0])
    Xind = X_test[np.array([ind])]
    y_predict = codec.dec(model.predict(Xind))
    y2 = model.predict_classes(Xind, verbose=0)
    y_true = codec.dec(yc_test[ind])
    print('y_true is ',y_true, '.learned machine predicts', y_predict, y2)

for _ in range(10):
    view_sample()