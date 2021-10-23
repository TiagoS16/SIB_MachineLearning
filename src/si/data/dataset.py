import numpy as np
import pandas as pd
from si.util.util import label_gen

__all__ = ['Dataset']


class Dataset:
    def __init__(self, X=None, Y=None,
                 xnames: list = None,
                 yname: str = None):
        """ Tabular Dataset"""
        if X is None:
            raise Exception("Trying to instanciate a Dataset without any data")
        self.X = X
        self.Y = Y
        self.xnames = xnames if xnames else label_gen(X.shape[1])
        self.yname = yname if yname else 'Y'

    @classmethod
    def from_data(cls, filename, sep=",", labeled=True):
        """Creates a DataSet from a data file.

        :param filename: The filename
        :type filename: str
        :param sep: attributes separator, defaults to ","
        :type sep: str, optional
        :return: A DataSet object
        :rtype: DataSet
        """
        data = np.genfromtxt(filename, delimiter=sep)
        if labeled:
            X = data[:, 0:-1]
            Y = data[:, -1]
        else:
            X = data
            Y = None
        return cls(X, Y)

    @classmethod
    def from_dataframe(cls, df, ylabel=None):
        """Creates a DataSet from a pandas dataframe.

        :param df: [description]
        :type df: [type]
        :param ylabel: [description], defaults to None
        :type ylabel: [type], optional
        :return: [description]
        :rtype: [type]
        """
        if ylabel is not None and ylabel in df.columns:
            X = df.loc[:, df.columns != ylabel].to_numpy()
            Y = df.loc[:, ylabel].to_numpy()
            xnames = df.columns.tolist().remove(ylabel)
            yname = ylabel

        else:
            X = df.to_numpy()
            Y = None
            xnames = df.columns.tolist()
            yname = None

        return cls(X, Y, xnames, yname)


    def __len__(self):
        """Returns the number of data points."""
        return self.X.shape[0]

    def hasLabel(self):
        """Returns True if the dataset constains labels (a dependent variable)"""
        return self.Y

    def getNumFeatures(self):
        """Returns the number of features"""
        return self.X.shape[1]

    def getNumClasses(self):
        """Returns the number of label classes or 0 if the dataset has no dependent variable."""
        return len(np.unique(self.Y)) if self.hasLabel() else 0

    def writeDataset(self, filename, sep=","):
        """Saves the dataset to a file

        :param filename: The output file path
        :type filename: str
        :param sep: The fields separator, defaults to ","
        :type sep: str, optional
        """

        fullds = np.hstack((self.X, self.Y.reshape(len(self.Y), 1)))
        np.savetxt(filename, fullds, delimiter=sep)

    def toDataframe(self):
        """ Converts the dataset into a pandas DataFrame"""
        df = pd.DataFrame(self.X, index=self.Y, columns=self.xnames)
        df.index.name = self.yname
        return df

    def getXy(self):
        return self.X, self.Y


def summary(dataset, format='df'):
    """ Returns the statistics of a dataset(mean, std, max, min)

    :param dataset: A Dataset object
    :type dataset: si.data.Dataset
    :param format: Output format ('df':DataFrame, 'dict':dictionary ), defaults to 'df'
    :type format: str, optional
    """

    if format not in ["df", "dict"]:
        raise Exception("Invalid format. Choose between 'df' and 'dict'.")

    if dataset.hasLabel():
        data = np.hstack([dataset.X, np.reshape(dataset.Y, (-1, 1))])
        columns = dataset.xnames[:] + [dataset.yname]
    else:
        data = dataset.X
        columns = dataset.xnames[:]

    _means = np.mean(data, axis=0)
    _vars = np.var(data, axis=0)
    _maxs = np.max(data, axis=0)
    _mins = np.min(data, axis=0)
    stats = {}
    for i in range(data.shape[1]):
        stat = {"mean": _means[i],
                "var": _vars[i],
                "max": _maxs[i],
                "min": _mins[i]
                }
        stats[columns[i]] = stat

    if format == "dict":
        return stats
    else:
        return pd.DataFrame.from_dict(stats, orient='index')

