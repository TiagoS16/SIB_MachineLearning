from .model import Model
from ..util.util import sigmoid
import numpy as np

class LogisticRegression(Model):

    def __init__(self, gd=False, epochs=1000, lr=0.1):
        super(LogisticRegression, self).__init__()
        self.gd = gd
        self.theta = None
        self.epochs = epochs
        self.lr = lr

    def fit(self, dataset):
        X, Y = dataset.getXy()
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        self.X = X
        self.Y = Y

        self.train(X, Y)
        self.is_fitted = True

    def train(self, x, y):
        n = x.shape[1]
        self.history = {}  # criar um historico dos thetas e custo por epoch
        self.theta = np.zeros(n)
        for epoch in range(self.epochs):
            z = np.dot(x, self.theta)
            h = sigmoid(z)
            grad = np.dot(x.T, (h - y)) / y.size
            self.theta -= self.lr * grad
            self.history[epoch] = [self.theta[:], self.cost()]

    def probability(self, x):
        assert self.is_fitted, 'Model must be fitted before predicting'
        _x = np.hstack(([1], x))
        return sigmoid(np.dot(self.theta, _x))

    def predict(self, x):
        p = self.probability(x)
        res = 1 if p >= 0.5 else 0
        return res

    def cost(self):
        h = sigmoid(np.dot(self.X, self.theta))
        cost = (-self.Y * np.log(h) - (1-self.Y) * np.log(1-h))
        res = np.sum(cost) / self.X.shape[0]
        return res


class LogisticRegressionReg(LogisticRegression):

    def __init__(self, gd=False, epochs=1000, lr=0.1, lbd=1):
        super(LogisticRegressionReg, self).__init__(gd=gd, epochs=epochs, lr=lr)
        self.lbd = lbd
