from abc import ABC, abstractmethod


class Modelo(ABC):
    def __init__(self, n_neighbors=5):
        self.is_fitted = False

    @abstractmethod
    def fit(self, dataset):
        raise NotImplementedError

    @abstractmethod
    def predict(self, x):
        raise NotImplementedError

    @abstractmethod
    def cost(self):
        raise NotImplementedError
