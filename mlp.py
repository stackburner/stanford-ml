# -*- coding: utf-8 -*-
"""
Multilayer (3) perceptron implementation for back-propagation learning. Implements
an artificial neural network trained for handwriting recognition using the MNIST
dataset.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist
from scipy.special import expit


def load_data(path, kind='train'):
    labels = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images = os.path.join(path, '%s-images-idx3-ubyte' % kind)
    images, labels = loadlocal_mnist(images_path=images, labels_path=labels)

    return images, labels


class MLP:

    def __init__(self, n_output, n_features, n_hidden=30, epochs=1000, eta=0.001):
        self.cost_ = []
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.w1, self.w2 = self._initialize_weights()
        self.epochs = epochs
        self.eta = eta


    def _initialize_weights(self):
        w1 = np.random.uniform(-1.0, 1.0, size=self.n_hidden * (self.n_features + 1))
        w1 = w1.reshape(self.n_hidden, self.n_features + 1)
        w2 = np.random.uniform(-1.0, 1.0, size=self.n_output * (self.n_hidden + 1))
        w2 = w2.reshape(self.n_output, self.n_hidden + 1)

        return w1, w2


    @staticmethod
    def _encode_labels(y, k):
        onehot = np.zeros((k, y.shape[0]))
        for i, val in enumerate(y):
            onehot[val, i] = 1.0

        return onehot


    def _feedforward(self, X, w1, w2):
        a1 = self._add_bias_unit(X, 'column')
        z2 = w1.dot(a1.T)
        a2 = self._activate(z2)
        a2 = self._add_bias_unit(a2, 'row')
        z3 = w2.dot(a2)
        a3 = self._activate(z3)

        return a1, z2, a2, z3, a3


    @staticmethod
    def _add_bias_unit(X, how):
        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0] + 1, X.shape[1]))
            X_new[1:, :] = X

        return X_new


    @staticmethod
    def _activate(z):
        sigmoid = expit(z)

        return sigmoid


    def fit(self, X, y):
        X_data, y_data = X.copy(), y.copy()
        y_enc = self._encode_labels(y, self.n_output)
        for i in range(self.epochs):
            """
            self.eta /= (1 + 0.00001 * i)
            idx = np.random.permutation(y_data.shape[0])
            X_data, y_enc = X_data[idx], y_enc[:, idx]
            """
            batches = np.array_split(range(y_data.shape[0]), 1)
            for batch in batches:
                a1, z2, a2, z3, a3 = self._feedforward(X_data[batch], self.w1, self.w2)
                """
                TODO:
                - compute cost
                - compute gradients
                """
        return self


X_train, y_train = load_data('mnist/', kind='train')
X_test, y_test = load_data('mnist/', kind='t10k')
nn = MLP(10, X_train.shape[1], 30, 100, 0.001)
nn.fit(X_train, y_train)
plt.plot(range(len(nn.cost_)), nn.cost_)
plt.ylim([0, 2000])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.tight_layout()
plt.show()
