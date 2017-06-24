# -*- coding: utf-8 -*-
"""objective: Learning addition. Input: "535+61", Expect output: "596" """
from __future__ import print_function
from keras.models import Sequential
from keras import layers
import numpy as np
from six.moves import range

class CharacterTable(object):
    # All the numbers, plus sign and space for padding.
    chars = sorted(set('0123456789+ '))

    def __init__(self):
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """One hot encode given string C. output num_rows by self.len matrix"""
        x = np.zeros((num_rows, self.len))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        """Decode a vector of probabilities to their character output"""
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)

    @property
    def len(self): return len(self.chars)

class colors:
    green, red, white = '\033[92m', '\033[91m', '\033[0m'

# Parameters for the model and dataset.
TRAINING_SIZE = 5000
DIGITS = 2

print(__doc__)


# Maximum length of input is 'int + int' (e.g., '345+678'). Maximum length of int is DIGITS.
MAXLEN = DIGITS + 1 + DIGITS

ctable = CharacterTable()

print('Generating questions...')
questions, expected = [], []
seen = set()
f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS + 1))))
while len(questions) < TRAINING_SIZE:
    a, b = f(), f()

    key = tuple(sorted((a, b)))
    if key in seen: continue # Skip q we've already seen, so are any such that x+Y == Y+x (hence the sorting).

    seen.add(key)
    # Pad the data with spaces such that it is always MAXLEN.
    q = '{}+{}'.format(a, b)
    query = q + ' ' * (MAXLEN - len(q))
    ans = str(a + b)
    # Answers can be of maximum size DIGITS + 1.
    ans += ' ' * (DIGITS + 1 - len(ans))
    # Reverse the query, e.g., '12+345  ' becomes '  543+21'.
    # (Note the space used for padding.)
    # query = query[::-1]
    questions.append(query)
    expected.append(ans)
print('Total addition questions:', len(questions))

print('Vectorization...')
x = np.zeros((TRAINING_SIZE, MAXLEN    , ctable.len), dtype=np.bool)
y = np.zeros((TRAINING_SIZE, DIGITS + 1, ctable.len), dtype=np.bool)
for i, sentence in enumerate(questions): x[i] = ctable.encode(sentence, MAXLEN)
for i, sentence in enumerate(expected):  y[i] = ctable.encode(sentence, DIGITS + 1)

# Shuffle (x, y) in unison as the later parts of x will almost all be larger digits.
indices = np.arange(len(y))
np.random.shuffle(indices)
x, y = x[indices], y[indices]

split_at = len(x) // 10 # 10% for validation
(x_val, x_train) = x[:split_at], x[split_at:]
(y_val, y_train) = y[:split_at], y[split_at:]
print('x_train', x_train.shape,'y_train',y_train.shape)
print('x_val', x_val.shape,'y_val',y_val.shape)


# Try replacing GRU, or SimpleRNN.
RNN = layers.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = TRAINING_SIZE // 10
LAYERS = 1

print('Build model...')
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
# Note: In a situation where your input sequences have a variable length,
# use input_shape=(None, num_feature).
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, ctable.len)))
# As the decoder RNN's input, repeatedly provide with the last hidden state of
# RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
# length of output, e.g., when DIGITS=3, max output is 999+999=1998.
model.add(layers.RepeatVector(DIGITS + 1))
# The decoder RNN could be multiple layers stacked or a single layer.
for _ in range(LAYERS):
    # By setting return_sequences to True, return not only the last output but
    # all the outputs so far in the form of (num_samples, timesteps,
    # output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

# Apply a dense layer to the every temporal slice of an input. For each of step
# of the output sequence, decide which character should be chosen.
model.add(layers.TimeDistributed(layers.Dense(ctable.len)))
model.add(layers.Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

for iteration in range(1, 2):
    print('Iteration', iteration)
    model.fit(x_train, y_train,validation_data=(x_val, y_val),
              batch_size=BATCH_SIZE, epochs=1, verbose=0)

    # Select 10 samples from the validation set at random so we can visualize errors.
    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        
        preds = model.predict_classes(rowx, verbose=0)
        guess = ctable.decode(preds[0], calc_argmax=False)

        print(colors.green if guess == correct else colors.red, 'Q:', q, ' == ', guess, colors.white, end="\n")
