# A Neural Network in 13 lines of Python
# https://iamtrask.github.io/2015/07/27/python-network-part2/

# 2 layers NN learns XOR

import numpy as np

sigmoid = lambda x : 1/(1 + np.exp(-x))
dsigmoid = lambda x : x*(1-x)
    
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
y = np.array([[0],[1],[1],[0]]) # XOR

np.random.seed(1)
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1


e1 = np.empty(shape = (4,1))
e2 = np.empty(shape = (4,1))
syn0_history = np.empty(shape = (12,1))
syn1_history = np.empty(shape = (4,1))

for j in range(60000):

    # Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = sigmoid(np.dot(l0,syn0))
    l2 = sigmoid(np.dot(l1,syn1))

    # how much did we miss the target value?
    l2_error = l2 - y
    l2_delta = l2_error*dsigmoid(l2)

    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * dsigmoid(l1)

    syn1 -= l1.T.dot(l2_delta)
    syn0 -= l0.T.dot(l1_delta)

    
    if (j% 10000) == 0:
        print("Error:", str(np.mean(np.abs(l2_error))))
        e1 = np.hstack([e1,l1_error])
        e2 = np.hstack([e2,l2_error])
        syn0_history = np.hstack([syn0_history, syn0.reshape([12,1])])
        syn1_history = np.hstack([syn1_history, syn1.reshape([4,1])])