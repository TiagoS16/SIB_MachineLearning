# Linear Regression
print('Linear Regression')

from src.si.data.dataset import Dataset, summary
from src.si.util.scale import StandardScaler
from src.si.supervised.linreg import LinearRegression, LinearRegressionReg
import numpy as np
import os


DIR = os.path.dirname(os.path.realpath('.'))
filename = os.path.join(DIR, 'datasets/lr-example1.data')
dataset = Dataset.from_data(filename, labeled=True)
StandardScaler().fit_transform(dataset, inline=True)
print(summary(dataset))

import matplotlib.pyplot as plt
if dataset.X.shape[1] == 1:
    plt.scatter(dataset.X, dataset.Y)
    plt.show()

## Linear Regression using closed form
print('\nLinear Regression using closed form')

lr = LinearRegression()
lr.fit(dataset)
print('Theta = ', lr.theta)

idx = 10
x = dataset.X[idx]
print("x = ", x)
y = dataset.Y[idx]
y_pred = lr.predict(x)
print("y_pred = ", y_pred)
print("y_true = ", y)

print('Custo:', lr.cost())

if dataset.X.shape[1] == 1:
    plt.scatter(dataset.X, dataset.Y)
    plt.plot(lr.X[:, 1], np.dot(lr.X, lr.theta), '-', color='red')
    plt.show()


## Linear Regression using gradient descent
print('\nLinear Regression using gradient descent')

lr = LinearRegression(gd=True,epochs=50000)
lr.fit(dataset)
print('Theta = ', lr.theta)

plt.plot(list(lr.history.keys()), [y[1] for y in lr.history.values()], '-', color='red')
plt.title('Cost')
plt.show()

# Linear Regression with Regularization
print('\nLinear Regression with Regularization')

lr = LinearRegressionReg()
lr.fit(dataset)
print('Theta = ', lr.theta)

idx = 10
x = dataset.X[idx]
print("x = ", x)
y = dataset.Y[idx]
y_pred = lr.predict(x)
print("y_pred = ", y_pred)
print("y_true = ", y)
print('Custo:', lr.cost())

# Logistic Regression
print('\nLogistic Regression')

from src.si.supervised.logreg import LogisticRegression, LogisticRegressionReg
import pandas as pd

filename = os.path.join(DIR, 'datasets/iris.data')
df = pd.read_csv(filename)
iris = Dataset.from_dataframe(df, ylabel="class")
y = [int(x != 'Iris-setosa') for x in iris.Y]
dataset = Dataset(iris.X[:, :2], np.array(y))
print(summary(dataset))

plt.scatter(dataset.X[:, 0], dataset.X[:, 1], c=dataset.Y)
plt.show()

logreg = LogisticRegression(epochs=20000)

logreg.fit(dataset)
print(logreg.theta)

plt.scatter(dataset.X[:, 0], dataset.X[:, 1], c=dataset.Y)
_x = np.linspace(min(dataset.X[:, 0]), max(dataset.X[:, 0]), 2)
_y = [(-logreg.theta[0] - logreg.theta[1] * x) / logreg.theta[2] for x in _x]
plt.plot(_x, _y, '-', color='red')
plt.show()

plt.plot(list(logreg.history.keys()), [y[1] for y in logreg.history.values()], '-', color='red')
plt.title('Cost')
plt.show()

ex = np.array([5.5, 2])
print("Pred. example:", logreg.predict(ex))
print('Custo:', logreg.cost())


# Logistic Regression with L2 regularization
print('\nLogistic Regression with L2 regularization')

# logreg = LogisticRegressionReg()
# logreg.fit(dataset)
# print(logreg.theta)
#
# plt.scatter(dataset.X[:, 0], dataset.X[:, 1], c=dataset.Y)
# _x = np.linspace(min(dataset.X[:, 0]), max(dataset.X[:, 0]), 2)
# _y = [(-logreg.theta[0]-logreg.theta[1]*x)/logreg.theta[2] for x in _x]
# plt.plot(_x, _y, '-', color='red')
# plt.show()
#
# ex = np.array([5.5, 2])
# print("Pred. example:", logreg.predict(ex))
# print('Custo:', logreg.cost())
