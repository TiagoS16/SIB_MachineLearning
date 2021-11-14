import warnings
import numpy as np
import scipy as stats

from src.si.data import Dataset
from src.si.util.util import euclidean, manhattan
from src.si.util.scale import StandardScaler


class PCA:
    def __init__(self, n_components=2, method="svd"):
        self.n_components = n_components

        available_methods = ["svd", "evd"]
        if method not in available_methods:
            raise Exception(f"Method not available. Please choose between: {available_methods}.")
        self.method = method

    def fit(self, dataset):
        pass

    def transform(self, dataset):
        x_scaled = StandardScaler().fit_transform(dataset)  # normalizar os dados
        features = x_scaled.X.T

        if self.method == "svd":
            self.vecs, self.vals, rv = np.linalg.svd(features)
        else:
            cov_matrix = np.cov(features)
            self.vals, self.vecs = np.linalg.eig(cov_matrix)

        self.sorted_idx = np.argsort(self.vals)[::-1]  # indices ordenados por importancia das componentes
        self.sorted_value = self.vals[self.sorted_idx]  # ordenar os valores pelos indices das colunas
        self.sorted_vectors = self.vecs[:, self.sorted_idx]  # ordenar os vetores pelos indices das colunas

        if self.n_components > 0:
            if self.n_components > dataset.X.shape[1]:
                warnings.warn("The number of components is larger than the number of features.")
                self.n_components = dataset.X.shape[1]
            self.components_vector = self.sorted_vectors[:, 0:self.n_components]  # vetores correspondentes ao numero de componentes selecionados
        else:
            warnings.warn("The number of components is lower than 0.")
            self.n_components = 1
            self.components_vector = self.sorted_vectors[:, 0:self.n_components]

        x_red = np.dot(self.components_vector.transpose(), features).transpose()
        return x_red

    def fit_transform(self, dataset):
        x_red = self.transform(dataset)
        components_sum = self.explained_variances()
        return x_red, components_sum

    def explained_variances(self):
        sum_val = np.sum(self.sorted_value)
        ev = []
        for value in self.sorted_value:
            ev.append(value / sum_val * 100)
        return np.array(ev)


class KMeans:
    def __init__(self, k, distance='euclidean', n_iter=100):  # todo verificar argumentos default (Enum - lista para selecionar a funçao)
        dist = ["euclidean", "manhattan"]
        self.k = k
        self.iter = n_iter
        self.centroids = None

        if distance not in dist:
            raise Exception(f"Distance selected is not present on the list of available functions: {dist}")
        elif distance == "euclidean":
            self.distance = euclidean  # func no script util.py
        else:
            self.distance = manhattan

    def fit(self, dataset):
        x = dataset.X
        self._min = np.min(x, axis=0)
        self._max = np.max(x, axis=0)

    def init_centroids(self, dataset):
        x = dataset.X
        # self.centroids = np.array([np.random.uniform(low=self._min[i], high=self._max[i], size=(self.k,))
        #                            for i in range(x.shape[1])]).T
        rng = np.random.default_rng()
        self.centroids = rng.choice(x, size=(self.k), replace=False, p=None, axis=0)

    def get_closest_centroid(self, x):
        dist = self.distance(x, self.centroids)
        closest_centroid_index = np.argmin(dist, axis=0)
        return closest_centroid_index

    def transform(self, dataset):
        self.init_centroids(dataset)
        x = dataset.X
        changed = False
        count = 0
        old_idxs = np.zeros(x.shape[0])
        while count < self.iter and not changed:
            idxs = np.apply_along_axis(self.get_closest_centroid, axis=0, arr=x.T)

            centroids = []
            for i in range(self.k):
                centroids.append(np.mean(x[idxs == i], axis=0))
            self.centroids = np.array(centroids)

            changed = np.all(old_idxs == idxs)
            # changed = np.array_equal(old_idxs, idxs)
            old_idxs = idxs

            count += 1
        return self.centroids, old_idxs

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)

