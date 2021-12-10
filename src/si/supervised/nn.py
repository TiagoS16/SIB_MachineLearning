import numpy as np
from abc import ABC, abstractmethod
from .model import Model
from ..util.activation import *
from ..util.metrics import mse


class Layer(ABC):

    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input):
        raise NotImplementedError

    @abstractmethod
    def backward(self, erro, learning_rate):
        raise NotImplementedError


class Dense(Layer):

    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size)  # -0.5
        self.bias = np.zeros((1, output_size))

    def setWeights(self, weights, bias):
        if (weights.shape != self.weights.shape):
            raise ValueError(f"Shapes mismatch {weights.shape} and {self.weights.shape}")
        if (bias.shape != self.bias.shape):
            raise ValueError(f"Shapes mismatch {bias.shape} and {self.bias.shape}")
        self.weights = weights
        self.bias = bias

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, erro, learning_rate):
        raise NotImplementedError


class Activation(Layer):

    def __init__(self, func):
        self.func =func

    def forward(self, input):
        self.input = input
        self.output = self.func(self.input)
        return self.output

    def backward(self, erro, learning_rate):
        raise NotImplementedError


class NN(Model):

    def __init__(self, epochs=1000, lr=0.01, verbose=True):
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose

        self.layers = []
        self.loss = mse
        # self.loss_prime = mse_prime

    def add(self, layer):
        self.layers.append(layer)

    def fit(self, dataset):
        self.dataset = dataset
        raise NotImplementedError

    def predict(self, x):
        assert self.is_fitted, "Model must be fitted before prediction"
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def cost(self, X=None, Y=None):
        assert self.is_fitted, "Model must be fitted before prediction"
        X = X if X is not None else self.dataset.X  # criar funçao de
        Y = Y if Y is not None else self.dataset.Y
        output = self.predict(X)
        return self.loss(Y, output)