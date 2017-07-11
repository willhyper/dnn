"""objective: Learning addition. Input: "53+61", Expect output: "114" """
from __future__ import print_function
from keras.models import Sequential
from keras import layers
import numpy as np

MAX_DIGIT = 2
TRAINING_SIZE = 5000 # < 10**(MAX_DIGIT*2)
STOP_ACCURACY = 0.9 # fit until this percentage

print(__doc__, '. Learning %d digit addition' % MAX_DIGIT, '. Fitting until accuracy %f' % STOP_ACCURACY)

class CharacterTable(object):
    SPACE = '_'
    def __init__(self, chars):
        self.chars = chars
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, s):
        """One hot encode given string s of len R. output R by C matrix."""
        x = np.zeros((len(s), len(self.chars)))
        for i, c in enumerate(s):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        """Decode a vector of probabilities to their character output"""
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)

    @property
    def len(self): return len(self.chars)

def as_len_n_str(n):
    s = str(n)
    return CharacterTable.SPACE * (MAX_DIGIT + 1 - len(s)) + s

def randint_generator(digits):
    exclusive_max = 10 ** digits
    while True:
        yield np.random.randint(low=0, high=exclusive_max)
gen_int = randint_generator(MAX_DIGIT)

print('Generating %d questions...' % TRAINING_SIZE)
questions = dict()
while len(questions) < TRAINING_SIZE:
    a, b = next(gen_int), next(gen_int)
    key = tuple(sorted((a, b)))
    if key not in questions:
        questions[key] = a + b

print('One hot encoding questions...')
ctable = CharacterTable(sorted(set('0123456789+' + CharacterTable.SPACE)))

input_timestep = 2 * MAX_DIGIT + 3 # when MAX_DIGIT=2, '_99+_99' is 7 char
output_timestep = MAX_DIGIT + 1 # when MAX_DIGIT=2, max output '198' is 3 char
n_features = ctable.len
x = np.zeros((TRAINING_SIZE, input_timestep, n_features), dtype=np.bool)
y = np.zeros((TRAINING_SIZE, output_timestep, n_features), dtype=np.bool)

for i, question in enumerate(questions.items()):
    (a, b), c = question
    q = '%s+%s' % (as_len_n_str(a), as_len_n_str(b))
    x[i] = ctable.encode(q)
    y[i] = ctable.encode(as_len_n_str(c))

split_at = len(x) // 10  # 10% for validation
(x_test, x_train) = x[:split_at], x[split_at:]
(y_test, y_train) = y[:split_at], y[split_at:]
print('x_train', x_train.shape, 'y_train', y_train.shape)
print('x_test', x_test.shape, 'y_test', y_test.shape)

print('Build model...')
RNN = layers.LSTM  # layers.GRU # layers.SimpleRNN
model = Sequential()
model.add(RNN(units=10, input_shape=(input_timestep, n_features))) # 10 is design parameter
model.add(layers.RepeatVector(output_timestep)) # Repeatedly provide with the last hidden state of RNN for each time step.
model.add(RNN(units=20, return_sequences=True)) # 20 is design parameter
model.add(layers.TimeDistributed(layers.Dense(n_features)))
model.add(layers.Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

def view_sample():
    class colors: green, red, white = '\033[92m', '\033[91m', '\033[0m'

    for _ in range(10):
        ind = np.random.randint(0, len(x_test))
        npi = np.array([ind])
        xi, yi = x_test[npi], y_test[npi]
        q, true_answer = ctable.decode(xi[0]), ctable.decode(yi[0])

        preds = model.predict_classes(xi, verbose=0)
        guess = ctable.decode(preds[0], calc_argmax=False)

        print(colors.green if guess == true_answer else colors.red, 'Q:', q, ' == ', guess, colors.white, end="\n")

while True:
    hist = model.fit(x_train, y_train, batch_size=TRAINING_SIZE // 10, epochs=5, verbose=0)
    acc_of_last_epoch = hist.history['acc'][-1]
    print('acc_of_last_epoch', acc_of_last_epoch)
    view_sample()
    if acc_of_last_epoch > STOP_ACCURACY : break

loss, acc = model.evaluate(x_test, y_test)
print('test_acc', acc)