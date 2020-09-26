# -*- coding: utf-8 -*-
"""
Multilayer perceptron implementation for forward propagation learning. Implements
an artificial neural network trained for handwriting recognition using the MNIST
dataset.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist


def load_data(path, kind='train'):
    labels = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images = os.path.join(path, '%s-images-idx3-ubyte' % kind)
    images, labels = loadlocal_mnist(images_path=images, labels_path=labels)

    return images, labels


class MLP:

    def __init__(self, epochs=500, alpha=0.001):
        self.epochs = epochs
        self.alpha = alpha

    def _activate(self, z):
        sigmoid = 1.0 / (1.0 + np.exp(-z))

        return sigmoid

    def fit(self, X, y):
        self.cost_ = []

        return self


X_train, y_train = load_data('mnist/', kind='train')
X_test, y_test = load_data('mnist/', kind='t10k')
nn = MLP(epochs=1000, alpha=0.001)
nn.fit(X_train, y_train)
plt.plot(range(len(nn.cost_)), nn.cost_)
plt.ylim([0, 2000])
plt.ylabel('Cost')
plt.xlabel('Epochs * 50')
plt.tight_layout()
plt.show()
