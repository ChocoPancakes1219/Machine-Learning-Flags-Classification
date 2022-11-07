import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

import warnings

warnings.filterwarnings('ignore')
n_folds = 5


def visualise(cm, ag, X, y):
    plot(cm)
    kf(ag, X, y)


def plot(cm):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(cm, cmap=plt.cm.Reds, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
    classes = ["Catholic", "Other Christian", "Muslim", "Buddhist", "Hindu", "Ethnic", "Marxist", "Others"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Values', )
    plt.ylabel('Actual Values');


def kf(ag, X, y):
    # K-fold cross validation
    scores = np.average(cross_val_score(ag, X, y, cv=n_folds))
    print('The {}-fold cross-validation accuracy score for this classifier is {:.2f} '.format(n_folds, scores.mean()))


data = pd.read_csv('flag.csv')

# Cleaning the data
le = LabelEncoder()
data['mainhue'] = le.fit_transform(data['mainhue'])
data['topleft'] = le.fit_transform(data['topleft'])
data['botright'] = le.fit_transform(data['botright'])

# Creating dummy data to classify string data
X = data.drop(['name','religion'], axis=1)
y = data['religion']

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

# Support Vector Machine
print('\n\nSupport Vector Machine')
clf = svm.SVC(C=2, kernel='rbf', degree=50, gamma='auto')
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
# Classification Report
print(classification_report(y_test, predictions))
# Evaluate
cm = confusion_matrix(y_test, predictions)
visualise(cm, clf, X, y)

# Multi Layer Perceptron
print('\n\nMulti Layer Perceptron')
mlp = MLPClassifier(hidden_layer_sizes=(6,4), solver='lbfgs', random_state=42, max_iter=50000)
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)
# Classification Report
print(classification_report(y_test, predictions))
# Evaluate
cm = confusion_matrix(y_test, predictions)
visualise(cm, mlp, X, y)

# Linear Regression
print('\n\nLinear Regression')
reg = LinearRegression()
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
# Classification Report
cutoff = 0.9
predictions_classes = np.zeros_like(predictions)
predictions_classes[predictions > cutoff] = 1
print(classification_report(y_test, predictions_classes))
# Evaluate
cm = confusion_matrix(y_test, predictions_classes)
visualise(cm, reg, X, y)

# Decision Tree
print('\n\nDecision Tree')
dt = DecisionTreeClassifier(class_weight='balanced')
dt.fit(X_train, y_train)
predictions = dt.predict(X_test)
# Classification Report
print(classification_report(y_test, predictions))
# Evaluate
cm = confusion_matrix(y_test, predictions)
visualise(cm, dt, X, y)

plt.show()