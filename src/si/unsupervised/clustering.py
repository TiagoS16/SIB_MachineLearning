import warnings
import numpy as np
import scipy as stats
from sklearn.preprocessing import StandardScaler

from src.si.data import Dataset
from src.si.util.util import euclidean, manhattan


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
        x = dataset.X

        x_scaled = StandardScaler().fit_transform(x)  # normalizar os dados
        features = x_scaled.T

        if self.method == "svd":
            self.vecs, self.vals, rv = np.linalg.svd(features)
        else:
            cov_matrix = np.cov(features)
            self.vals, self.vecs = np.linalg.eig(cov_matrix)

        self.sorted_idx = np.argsort(self.vals)[::-1]  # indices ordenados por importancia das componentes
        self.sorted_e_value = self.vals[self.sorted_idx]  # ordenar os valores pelos indices das colunas
        self.sorted_e_vectors = self.vecs[:, self.sorted_idx]  # ordenar os vetores pelos indices das colunas

        if self.n_components > 0:
            if self.n_components > x.shape[1]:
                warnings.warn("The number of components is larger than the number of features.")
                self.n_components = x.shape[1]
            self.components_vector = self.sorted_e_vectors[:, 0:self.n_components]  # vetores correspondentes ao numero de componentes selecionados
        else:
            warnings.warn("The number of components is lower than 0.")
            self.n_components = 1
            self.components_vector = self.sorted_e_vectors[:, 0:self.n_components]

        x_red = np.dot(self.components_vector.transpose(), features).transpose()
        return x_red

    def fit_transform(self, dataset):
        x_red = self.transform(dataset)
        components_sum, components_values = self.explained_variances()
        return x_red, components_sum, components_values

    def explained_variances(self):
        self.components_values = self.sorted_e_value[0:self.n_components] / np.sum(self.sorted_e_value)
        return np.sum(self.components_values), self.components_values


class KMeans:
    def __init__(self, k, distance, n_iter=100):
        dist = ["euclidean", "manhattan"]
        self.k = k
        self.iter = n_iter
        self.centroids = None

        if distance not in dist:
            raise Exception(f"Distance selected is not present on the list of available functions: {dist}")
        elif distance is "euclidean":
            self.distance = euclidean  # func no script util.py
        else:
            self.distance = manhattan

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

            centroids = []
            for i in range(self.k):
                centroids.append(np.mean(x[idxs == i], axis=0))
            self.centroids = np.array(centroids)

            changed = np.all(old_idxs == idxs)
            old_idxs = idxs

            count += 1
        return self.centroids, old_idxs

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)

