from si.data.dataset import *
from si.data.feature_selection import *
from si.util.util import *
from si.util.scale import *
from si.unsupervised.clustering import *
import pandas as pd
import matplotlib.pyplot as plt
import os

DIR = os.path.dirname(os.path.realpath('.'))
filename = os.path.join(DIR, 'datasets/breast-bin.data')

# Labeled dataset
print("Labeled dataset")

dataset = Dataset.from_data(filename, labeled=True)

print(dataset.X[:5, :])

print(dataset.Y[:5])

print("Has label:", dataset.hasLabel())
print("Number of features:", dataset.getNumFeatures())
print("Number of classes:", dataset.getNumClasses())
print(summary(dataset))

dataset.toDataframe()

# Standard Scaler
print("\nStandard Scaler")

sc = StandardScaler()
ds2 = sc.fit_transform(dataset)
print(summary(ds2))

# Feature Selection

# Variance Threshold
print('\nVariance Threshold')

vt = VarianceThreshold(8)
ds2 = vt.fit_transform(dataset)
print(summary(ds2))

# SelectKBest
print('\nSelectKBest')

skb = SelectKBest(5)
ds3 = skb.fit_transform(dataset)
print(summary(ds3))

# Clustering

filename = os.path.join(DIR, 'datasets/iris.data')
df = pd.read_csv(filename)
iris = Dataset.from_dataframe(df, ylabel="class")

# indice das features para o plot
c1 = 0
c2 = 1
# plot
# plt.scatter(iris.X[:, c1], iris.X[:, c2])
# plt.xlabel(iris.xnames[c1])
# plt.ylabel(iris.xnames[c2])
# plt.show()

# KMeans
print('\nKMeans')

kmeans = KMeans(3)
cent, clust = kmeans.fit_transform(iris)
print(cent)

# plt.scatter(iris.X[:, c1], iris.X[:, c2], c=clust)
# plt.scatter(cent[:, c1], cent[:, c2], s=100, c='black', marker='x')
# plt.xlabel(iris.xnames[c1])
# plt.ylabel(iris.xnames[c2])
# plt.show()

# PCA
print('\nPCA')

pca = PCA(2, method='svd')

reduced = pca.fit_transform(iris)[0]
print(pca.explained_variances())

iris_pca = Dataset(reduced, iris.Y, xnames=['pc1', 'pc2'], yname='class')
iris_pca.toDataframe()

# plt.scatter(iris_pca.X[:, 0], iris_pca.X[:, 1])
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.show()

# KNN
print('\nKNN')
from src.si.supervised.knn import KNN

knn = KNN(3)
knn.fit(iris)

idx = 50
x = iris.X[idx, :]
pred = knn.predict(x)
print(f'Prediction: {pred}')
print(f'True: {iris.Y[idx]}')
print(knn.cost())

# SPLIT
print('\nSPLIT')
train, test = train_test_split(iris)
knn.fit(train)
idx = 15
x = test.X[idx, :]
pred = knn.predict(x)
print(f'Prediction: {pred}')
print(f'True: {test.Y[idx]}')

print(summary(train))
print()
print(summary(test))


# NB
print('\nNaive Bayes')
from src.si.supervised.nb import Naive_bayes

nb = Naive_bayes()
nb.fit(iris)

idx = 50
x = iris.X[idx, :]
pred = nb.predict(x)
print(f'Prediction: {pred}')
print(f'True: {iris.Y[idx]}')
print(nb.cost())
