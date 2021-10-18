import numpy as np
import scipy as stats
from copy import copy
import warnings

class VarianceThreshold:

    def __init__(self, threshold=0):
        if threshold < 0:
            warnings.warn("The threshold is a non-negative value.")
        self.threshold = threshold

    def fit(self, dataset):
        X = dataset.X
        self._var = np.var(X, axis=0)

    def tansform(self, dataset, inline=False):
        X = dataset.X
        cond = self._var > self.threshold
        idxs = [i for i in range(len(cond)) if cond[i]]
        X_trans = X[:, idxs]
        xnames = [dataset._xnames[i] for i in idxs]
        if inline:
            dataset.X = X_trans
            dataset._xnames = xnames
            return dataset
        else:
            from .dataset import Dataset
            return Dataset(copy(X_trans))
            # todo descobrir isto

    def fit_transform(self, dataset):
        # todo acabar isto
        pass

class SelectKBest:

    # todo class SelectKBest (scikitlearn) seleciona as KBest features