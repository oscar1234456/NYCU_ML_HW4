import numpy as np


def gradient_descent(phi, t):
    # init
    w = np.random.rand(3, 1)
    new_w = None
    MAX_Iter = 10000
    eps = 0.001
    learning_rate = 0.001
    for now_iter in range(MAX_Iter):
        # eps
        gradient = phi.T @ (1 / (1 + np.exp(-phi @ w)) - t)
        new_w = w - learning_rate * gradient
        if np.allclose(new_w, w, rtol=eps):
            print("__converge! early stop!__")
            return new_w
        w = new_w
    return w
