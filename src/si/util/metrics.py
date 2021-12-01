import numpy as np
import pandas as pd


def accuracy_score(y_true, y_pred):
    correct = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1
    accuracy = correct / len(y_true)
    return accuracy


def mse(y_true, y_pred, squared=True):
    """
    mean squared error
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    errors = np.average((y_true - y_pred) ** 2, axis=0)
    if not squared:
        errors = np.sqrt(errors)
    return np.average(errors)


class ConfusionMatrix:
    def __init__(self, true_y, pred_y):
        self.true_y = np.array(true_y)
        self.pred_y = np.array(pred_y)
        self.conf = None


    def calc(self):
        self.conf = pd.crosstab(self.true_y, self.pred_y, rownames=['Actual'], colnames=['Predicted'], margins=True)

    def toDataframe(self):
        return self.conf

    def __call__(self):
        self.calc()
