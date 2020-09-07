# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def calc_cost(thetas, X, y):
    m = len(y)
    predictions = np.dot(X, thetas)
    cost = (1/2*m) * np.sum(np.square(predictions-y))

    return cost


def gradient_descent(X, y, alpha=0.1, iterations=100000):
    m = len(y)
    thetas = np.random.randn(2, 1)
    cost = np.zeros(iterations)
    for i in range(iterations):
        prediction = np.dot(X, thetas)
        thetas = thetas - alpha * (1/m) * np.dot(np.transpose(X), (prediction-y))
        cost[i] = calc_cost(thetas, X, y)

    return thetas, cost


def normal_equation(X, y):
    thetas_ne = np.dot(np.linalg.inv(np.dot(np.transpose(X),X)),
                       np.dot(np.transpose(X), y))

    return thetas_ne


def plot_cost(cost):
    plt.plot(cost)
    plt.show()


def plot_fit(X, y, thetas):
    plt.plot(X, y, 'o')
    plt.plot(X, thetas[1]*X + thetas[0])
    plt.show()


X = 5 * np.random.rand(100, 1)
y = 1.5 + 4 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((len(X), 1)), X]
thetas, cost = gradient_descent(X_b, y)
print("Result using gradient descent:")
print(thetas)
thetas_ne = normal_equation(X_b, y)
print("Confirming result using normal equation:")
print(thetas_ne)
plot_cost(cost)
plot_fit(X, y, thetas)
