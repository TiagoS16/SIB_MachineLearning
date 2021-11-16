import itertools
import numpy as np
import pandas as pd

# Y is reserved to idenfify dependent variables
ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXZ'

__all__ = ['label_gen', 'euclidean', 'manhattan', 'sigmoid', 'train_test_split']


def label_gen(n):
    """ Generates a list of n distinct labels similar to Excel"""
    def _iter_all_strings():
        size = 1
        while True:
            for s in itertools.product(ALPHA, repeat=size):
                yield "".join(s)
            size += 1

    generator = _iter_all_strings()

    def gen():
        for s in generator:
            return s

    return [gen() for _ in range(n)]


def euclidean(x, y):
    dist = np.sqrt(np.sum((x - y)**2, axis=1))  # x is single point; y is various points
    return dist


def manhattan(x, y):
    dist = np.sum(np.abs(x - y))
    return dist


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train_test_split(dataset, split=0.8):
    size = dataset.X.shape[0]
    idx_split = int(split * size)
    arr = np.arange(size)
    np.random.shuffle(arr)
    from src.si.data.dataset import Dataset
    train = Dataset(dataset.X[arr[:idx_split]], dataset.Y[arr[:idx_split]], dataset.xnames, dataset.yname)
    test = Dataset(dataset.X[arr[idx_split:]], dataset.Y[arr[idx_split:]], dataset.xnames, dataset.yname)
    return train, test
