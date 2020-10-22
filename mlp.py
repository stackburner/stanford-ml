# -*- coding: utf-8 -*-
"""
Multilayer (3) perceptron implementation for back-propagation learning. Implements
an artificial neural network for handwriting recognition using the MNIST
dataset.
"""
import numpy as np
from scipy.special import expit


class MLP:

    def __init__(self, n_features, epochs, n_hidden=1000,
                 n_output=10, l1=0, l2=0.1, eta=0.001,
                 alpha=0.001, minibatches=50):
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.w1, self.w2 = self._initialize_weights()
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.minibatches = minibatches


    @staticmethod
    def _encode_labels(y, k):
        onehot = np.zeros((k, y.shape[0]))
        for i, val in enumerate(y):
            onehot[val, i] = 1.0

        return onehot


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


    @staticmethod
    def _L1_reg(lambda_, w1, w2):
        reg1 = (lambda_/2.0) * (np.abs(w1[:, 1:]).sum() + np.abs(w2[:, 1:]).sum())

        return reg1


    @staticmethod
    def _L2_reg(lambda_, w1, w2):
        reg2 = (lambda_/2.0) * (np.sum(w1[:, 1:] ** 2) + np.sum(w2[:, 1:] ** 2))

        return reg2


    def _activate_gradient(self, z):
        sg = self._activate(z)

        return sg * (1.0 - sg)


    def _initialize_weights(self):
        w1 = np.random.uniform(-1.0, 1.0, size=self.n_hidden * (self.n_features + 1))
        w1 = w1.reshape(self.n_hidden, self.n_features + 1)
        w2 = np.random.uniform(-1.0, 1.0, size=self.n_output * (self.n_hidden + 1))
        w2 = w2.reshape(self.n_output, self.n_hidden + 1)

        return w1, w2


    def _feedforward(self, X, w1, w2):
        a1 = self._add_bias_unit(X, 'column')
        z2 = w1.dot(a1.T)
        a2 = self._activate(z2)
        a2 = self._add_bias_unit(a2, 'row')
        z3 = w2.dot(a2)
        a3 = self._activate(z3)

        return a1, z2, a2, z3, a3


    def _calc_cost(self, y, theta, w1, w2):
        cost = np.sum(-y * (np.log(theta)) - (1.0 - y) * np.log(1.0 - theta))
        reg = self._L1_reg(self.l1, w1, w2) + self._L2_reg(self.l2, w1, w2)
        cost = cost + reg

        return cost


    def _backpropagate(self, a1, a2, a3, z2, y_enc, w1, w2):
        sigma3 = a3 - y_enc
        z2 = self._add_bias_unit(z2, how='row')
        sigma2 = w2.T.dot(sigma3) * self._activate_gradient(z2)
        sigma2 = sigma2[1:, :]
        grad1 = sigma2.dot(a1)
        grad2 = sigma3.dot(a2.T)
        grad1[:, 1:] += self.l2 * w1[:, 1:]
        grad1[:, 1:] += self.l1 * np.sign(w1[:, 1:])
        grad2[:, 1:] += self.l2 * w2[:, 1:]
        grad2[:, 1:] += self.l1 * np.sign(w2[:, 1:])

        return grad1, grad2


    def predict(self, X):
        a1, z2, a2, z3, a3 = self._feedforward(X, self.w1, self.w2)
        y_pred = np.argmax(z3, axis=0)

        return y_pred


    def fit(self, X, y):
        self.cost_ = []
        X_data, y_data = X.copy(), y.copy()
        y_enc = self._encode_labels(y, self.n_output)
        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)
        for i in range(self.epochs):
            self.eta /= (1 + 0.00001 * i)
            idx = np.random.permutation(y_data.shape[0])
            X_data, y_enc = X_data[idx], y_enc[:, idx]
            batches = np.array_split(range(y_data.shape[0]), self.minibatches)
            for batch in batches:
                # Forward propagation
                a1, z2, a2, z3, a3 = self._feedforward(X_data[batch], self.w1, self.w2)
                # Cost calculation
                cost = self._calc_cost(y_enc[:, batch], a3, self.w1, self.w2)
                self.cost_.append(cost)
                # Backward propagation
                grad1, grad2 = self._backpropagate(a1, a2, a3, z2, y_enc[:, batch], self.w1, self.w2)
                # Weight updates
                delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2
                self.w1 -= (delta_w1 + (self.alpha * delta_w1_prev))
                self.w2 -= (delta_w2 + (self.alpha * delta_w2_prev))
                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2

        return self
