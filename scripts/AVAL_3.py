# Ensemble (voting)

from src.si.data.dataset import Dataset, summary
from src.si.util.cv import CrossValidationScore
import os

DIR = os.path.dirname(os.path.realpath('.'))
filename = os.path.join(DIR, 'datasets/breast-bin.data')
dataset = Dataset.from_data(filename)
print(summary(dataset))

# Use accuracy as scorring function
from si.util import accuracy_score

### Decision Tree

from src.si.supervised.dt import DecisionTree

dt = DecisionTree()

cv = CrossValidationScore(dt, dataset, score=accuracy_score)
cv.run()
print(cv.toDataframe())

### Logistic regression

from src.si.supervised.logreg import LogisticRegression
logreg = LogisticRegression()

cv = CrossValidationScore(logreg, dataset, score=accuracy_score)
cv.run()
print(cv.toDataframe())

### KNN

from src.si.supervised.knn import KNN

knn = KNN(7)

cv = CrossValidationScore(knn, dataset, score=accuracy_score)
cv.run()
print(cv.toDataframe())

## Ensemble

def fvote(preds):
    return max(set(preds), key=preds.count)

from src.si.supervised.ensemble import Ensemble
en = Ensemble([dt, logreg, knn], fvote, accuracy_score)

cv = CrossValidationScore(en, dataset, score=accuracy_score)
cv.run()
print(cv.toDataframe())

## Confusion Matrix
from si.util.metrics import ConfusionMatrix

cm = ConfusionMatrix(cv.true_Y, cv.pred_Y)
cm()
print(cm.toDataframe())



