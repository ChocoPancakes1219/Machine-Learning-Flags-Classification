import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn import svm

data = pd.read_csv('flag.csv')

# Cleaning the data
data = data.drop('name', axis=1)
le = LabelEncoder()
data['mainhue'] = le.fit_transform(data['mainhue'])
data['topleft'] = le.fit_transform(data['topleft'])
data['botright'] = le.fit_transform(data['botright'])

# Separating the data
X = data.iloc[:, 0:-1].values
y = data. iloc[:, 1].values

# Data train, predict and evaluation

# SVM
kf = KFold(n_splits=5)
clf = svm.SVC(class_weight = 'balanced')
print("SVM 5-fold Scores")
for train_indices, test_indices in kf.split(X):
    clf.fit(X[train_indices], y[train_indices])
    print(clf.score(X[test_indices], y[test_indices]))

print("\n")

# MLP
kf2 = KFold(n_splits=5)
mlp = MLPClassifier(hidden_layer_sizes=(6, 4), solver='lbfgs', random_state=1, max_iter=50000)
print("MLP 5-fold Scores")
for train_indices, test_indices in kf2.split(X):
    mlp.fit(X[train_indices], y[train_indices])
    print(mlp.score(X[test_indices], y[test_indices]))

