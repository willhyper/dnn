'''
objective: 2 layers NN learns XOR
also demo the implementation of back propagation in numpy
'''
print(__doc__)

import numpy as np

sigmoid = lambda x: 1 / (1 + np.exp(-x))
dsigmoid = lambda x: x * (1 - x)

X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([[0, 1, 1, 0]]).T

n_sample, input_dim = X.shape
n_sample, output_dim = y.shape
n_neuron = 4  # design param

w0 = np.random.random((input_dim, n_neuron))
w2 = np.random.random((n_neuron, output_dim))

while True:
    l1 = np.dot(X, w0)  # Nxd = Nx3 * 3xd # N: number of sample, d = n_neuron, design param, 3 = input_dim
    l2 = sigmoid(l1)    # Nxd
    l3 = np.dot(l2, w2) # Nx1 = Nxd * dx1 , 1 = output_dim
    l4 = sigmoid(l3)    # Nx1

    if all((l4 > 0.5).astype(bool) == y.astype(bool)): break

    l4_error = l4 - y  # Nx1
    l3_error = l4_error * dsigmoid(l4)  # Nx1 = Nx1 .* Nx1
    l2_error = l3_error.dot(w2.T)  # Nxd = Nx1 * 1xd
    l1_error = l2_error * dsigmoid(l2)  # Nxd

    w2 -= l2.T.dot(l3_error)  # dx1 = dxN * Nx1
    w0 -= X.T.dot(l1_error)  # 3xd = 3xN * Nxd

print('last layer output, required by XOR', l4)