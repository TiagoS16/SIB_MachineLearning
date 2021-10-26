import numpy as np
import scipy as stats
from sklearn.pre

from src.si.data import Dataset
from src.util.util import euclidean

class PCA:
    def __init__(self, n_components=2, using="svd"):
        self.n_components = n_components
        self.using = using

    def fit(self, dataset):
        pass

    def transform(self, dataset):
        x = dataset.X
        n, p = x.shape

        scale =



class KMeans:

    def __init__(self, k, n_iter=100):
        self.k = k
        self.iter = n_iter
        self.centroids = None
        self.distance = euclidean  # func no script util.py

    def fit(self, dataset):
        x = dataset.X
        self._min = np.min(x, axis=0)
        self._max = np.max(x, axis=0)

    def init_centroids(self, dataset):
        x = dataset.X
        self.centroids = np.array([np.random.uniform(low=self._min[i], high=self._max[i], size=(self.k,))
                                   for i in range(x.shape[1])]).T

    def get_closest_centroid(self, x):
        dist = self.distance(x, self.centroids)
        closest_centroid_index = np.argmin(dist, axis=0)
        return closest_centroid_index

    def transform(self, dataset):
        self.init_centroids(dataset)
        print(self.centroids)
        x = dataset.X
        changed = True
        count = 0
        old_idxs = np.zeros(x.shape[0])
        while changed or count < self.iter:
            idxs = np.apply_along_axis(self.get_closest_centroid, axis=0, arr=x.T)

            cent = []
            for i in range(self.k):
                cent.append(np.mean(x[idxs == i], axis=0))
            self.centroids = np.array(cent)

            changed = np.all(old_idxs == idxs)
            old_idxs = idxs

            count += 1
        return self.centroids, old_idxs

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)

