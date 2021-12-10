
# Neural Networks

## XNOR

import numpy as np
from si.data import Dataset
X = np.array([[0, 0], [0, 1], [1,0], [1, 1]])
y = np.array([[1], [0], [0], [1]])
dataset = Dataset(X, y, ['X1', 'X2'], 'X1 XNOR X2')

print(dataset.toDataframe())


from si.supervised.nn import NN, Dense, Activation
from si.util.metrics import cross_entropy, cross_entropy_prime
from si.util.activation import Sigmoid

# layer1 weights
w1 = np.array([[20, -20], [20, -20]])
b1 = np.array([[-30, 10]])
l1 = Dense(2, 2)
l1.setWeights(w1, b1)

# layer2 weights
w2 = np.array([[20], [20]])
b2 = np.array([[-10]])
l2 = Dense(2, 1)
l2.setWeights(w2, b2)

# Build the NN
nn = NN()
nn.add(l1)
nn.add(Activation(Sigmoid()))
nn.add(l2)
nn.add(Activation(Sigmoid()))
# pass the info that the model is fitted
nn.is_fitted = True

print(np.round(nn.predict(dataset.X)))

print(nn.cost(dataset.X, dataset.Y))

#
# ## Train the model using backpropagation
#
# nn2 = NN(epochs=100000,lr=0.1, verbose=False)
# nn2.add(Dense(2, 2))
# nn2.add(Activation(Sigmoid()))
# nn2.add(Dense(2, 1))
# nn2.add(Activation(Sigmoid()))
# # use cross entropy instead of MSE as loss function
# # nn2.use(cross_entropy,cross_entropy_prime)
#
# nn2.fit(dataset)
#
# np.round(nn2.predict(X))
#
# import matplotlib.pyplot as plt
# plt.plot(list(nn2.history.keys()), list(nn2.history.values()), '-', color='red')
# plt.title('Loss')
# plt.show()
#
# np.round(nn2.predict(dataset.X))
#
# nn2.cost(dataset.X, dataset.y)
