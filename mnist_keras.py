from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.utils import to_categorical

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Data type:", X_train.dtype)

# Exercise 1:
# Reshape your data so that each image becomes a long vector
N_train = X_train.shape[0]
N_test = X_test.shape[0]
X_train = X_train.reshape(N_train, 28*28)
X_test = X_test.reshape(N_test, 28*28)

# Exercise 2:
# change the type of the input vectors to be 'float32'
# and rescale them so that the values are between 0 and 1
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0

# Exercise 3:
# convert class vectors to binary class matrices
yc_train = to_categorical(y_train, 10)
yc_test = to_categorical(y_test, 10)


# Exercise 4:
# https://keras.io/getting-started/sequential-model-guide/
# Choose your architecture as you please
model = Sequential()
model.add(Dense(20, input_dim = 784, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
model.summary()

# Exercise 5:
# Compile your model using an optimizer of your choice
# make sure to display the accuracy metric
# https://keras.io/optimizers/
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Exercise 6:
model.fit(X_train, yc_train, validation_split = 0.3)


# Exercise 7:
model.evaluate(X_test, yc_test)


# Bonus Exercise:
# Modify the code to use a Convolutional Neural Network
# Hints: you'll have to reshape your data to a 4D-array...