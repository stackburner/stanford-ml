#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training the mlp, dumping the trained weights in pickle objects and
validating the test- and training-data prediction accuracy. Optimized
for server-side runs (you may need to chmod +x the file for nohup
background processing the training).
"""
import os
from sys import platform
from mlxtend.data import loadlocal_mnist
import pickle
from mlp import *


def load_data(path, kind='train'):
    labels = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images = os.path.join(path, '%s-images-idx3-ubyte' % kind)
    images, labels = loadlocal_mnist(images_path=images, labels_path=labels)

    return images, labels


X_train, y_train = load_data('mnist/', kind='train')
X_test, y_test = load_data('mnist/', kind='t10k')

neural_net = MLP(X_train.shape[1], 100000)
neural_net.fit(X_train, y_train)

f = open('w1.pckl', 'wb')
pickle.dump(neural_net.w1, f)
f.close()
f = open('w2.pckl', 'wb')
pickle.dump(neural_net.w2, f)
f.close()

y_train_pred = neural_net.predict(X_train)
train_acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
y_test_pred = neural_net.predict(X_test)
test_acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
print('Training accuracy: %.2f%%' % (train_acc * 100))
print('Test accuracy: %.2f%%' % (test_acc * 100))

if platform != 'linux':
    import matplotlib.pyplot as plt
    plt.plot(range(len(neural_net.cost_)), neural_net.cost_)
    plt.ylabel('Cost')
    plt.xlabel('Epochs')
    plt.tight_layout()
    plt.show()
