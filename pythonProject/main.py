import numpy as np
import tensorflow as tf
import math
import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('flag.csv')

# Cleaning the data
le = LabelEncoder()
data['mainhue'] = le.fit_transform(data['mainhue'])
data['topleft'] = le.fit_transform(data['topleft'])
data['botright'] = le.fit_transform(data['botright'])

# Creating dummy data to classify string data
X = data.drop('name', axis=1)
y = data['religion']

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.6, random_state=42, stratify=y)

# SVM

clf = svm.SVC(C=1.0, kernel='rbf', degree=5, gamma='auto')
clf.fit(X_train, y_train)

# Classification Report

predictions = clf.predict(X_test)
print(classification_report(y_test, predictions))

#Confusion Matrix

cm = confusion_matrix(y_test, predictions)
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(cm, cmap=plt.cm.Reds, alpha=0.3)
for i in range(cm.shape[0]):
 for j in range(cm.shape[1]):
     ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
classes=["Catholic", "Other Christian", "Muslim", "Buddhist", "Hindu", "Ethnic", "Marxist", "Others"]
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted Values', )
plt.ylabel('Actual Values');

#K-fold Cross Validation

print('\n')
n_folds = 5
cv_error = np.average(cross_val_score(clf, X, y, cv=n_folds))
print('The {}-fold cross-validation accuracy score for this classifier is {:.2f}'.format(n_folds, cv_error))

#MLP
mlp = MLPClassifier(hidden_layer_sizes=(6, 4), solver='lbfgs', random_state=42, max_iter=50000)
mlp.fit(X_train, y_train)

# Classification Report

predictions = mlp.predict(X_test)
print(classification_report(y_test, predictions))

#Confusion Matrix

cm = confusion_matrix(y_test, predictions)
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(cm, cmap=plt.cm.Reds, alpha=0.3)
for i in range(cm.shape[0]):
 for j in range(cm.shape[1]):
     ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
classes=["Catholic", "Other Christian", "Muslim", "Buddhist", "Hindu", "Ethnic", "Marxist", "Others"]
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted Values', )
plt.ylabel('Actual Values');

#K-fold Cross Validation

print('\n')
n_folds = 5
cv_error = np.average(cross_val_score(mlp, X, y, cv=n_folds))
print('The {}-fold cross-validation accuracy score for this classifier is {:.2f}'.format(n_folds, cv_error))

#LR
reg = LinearRegression()
reg.fit(X_train, y_train)

# Classification Report

predictions = reg.predict(X_test)
cutoff = 0.9
predictions_classes = np.zeros_like(predictions)
predictions_classes[predictions > cutoff] = 1
print(classification_report(y_test, predictions_classes))

#Confusion Matrix

cm = confusion_matrix(y_test, predictions_classes)
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(cm, cmap=plt.cm.Reds, alpha=0.3)
for i in range(cm.shape[0]):
 for j in range(cm.shape[1]):
     ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
classes=["Catholic", "Other Christian", "Muslim", "Buddhist", "Hindu", "Ethnic", "Marxist", "Others"]
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted Values', )
plt.ylabel('Actual Values');

print('\n')
n_folds = 5
cv_error = np.average(cross_val_score(reg, X, y, cv=n_folds))
print('The {}-fold cross-validation accuracy score for this classifier is {:.2f}'.format(n_folds, cv_error))

#DT
dt = DecisionTreeClassifier(class_weight = 'balanced')
dt.fit(X_train, y_train)

# Classification Report

predictions = dt.predict(X_test)
print(classification_report(y_test, predictions))

#Confusion Matrix

cm = confusion_matrix(y_test, predictions)
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(cm, cmap=plt.cm.Reds, alpha=0.3)
for i in range(cm.shape[0]):
 for j in range(cm.shape[1]):
     ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
classes=["Catholic", "Other Christian", "Muslim", "Buddhist", "Hindu", "Ethnic", "Marxist", "Others"]
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted Values', )
plt.ylabel('Actual Values');

print('\n')
n_folds = 5
cv_error = np.average(cross_val_score(dt, X, y, cv=n_folds))
print('The {}-fold cross-validation accuracy score for this classifier is {:.2f}'.format(n_folds, cv_error))
