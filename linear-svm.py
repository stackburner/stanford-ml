# -*- coding: utf-8 -*-
"""
Fits a SVM with linear kernel to plot different decision boundaries based on
different values for parameter 'C'.
"""
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import scipy.io


def load_2dm_data(path):
    data = scipy.io.loadmat(path)
    X = np.array(data['X'])
    y = np.array([val for sublist in data['y'] for val in sublist])

    return X, y


def plot_data(X, y):
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='b', marker='x', label='1')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='r', marker='s', label='0')
    plt.tight_layout()
    plt.show()


def make_meshgrid(x, y, h=.01):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    return xx, yy


def make_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)

    return out


def plot_decision_boundary(X, y, svm):
    fig, ax = plt.subplots()
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    make_contours(ax, svm, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='b', marker='x', label='1')
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='r', marker='s', label='0')
    plt.show()


X, y = load_2dm_data('matlab/machine-learning-ex/ex6/ex6data1.mat')
plot_data(X, y)
svm = SVC(kernel='linear', C=1, random_state=0)
svm.fit(X, y)
plot_decision_boundary(X, y, svm)
svm = SVC(kernel='linear', C=100, random_state=0)
svm.fit(X, y)
plot_decision_boundary(X, y, svm)
