import itertools
import numpy as np
import pandas as pd

# Y is reserved to idenfify dependent variables
ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXZ'

__all__ = ['label_gen', 'euclidean', 'manhattan']


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
    dist = np.abs(x - y)
    dist = np.sum(dist)
    return dist
