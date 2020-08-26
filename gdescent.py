# -*- coding: utf-8 -*-
import numpy as np

theta1_true = 0.3
x = np.linspace(-1, 1, 10)
y = theta1_true * x


def calc_cost(x, y, theta1):
    m = len(y)
    total = 0
    for i in range(m):
        squared_error = (y[i] - theta1 * x) ** 2
        total += squared_error

    return total * (1 / (2 * m))


N = 50
alpha = 1
theta1 = [0]
J = [calc_cost(x, y, 0)[0]]
for j in range(N-1):
    last_theta1 = theta1[-1]
    this_theta1 = last_theta1 - alpha / len(y) * np.sum((x * last_theta1 - y) * x)
    theta1.append(this_theta1)


print(theta1)
