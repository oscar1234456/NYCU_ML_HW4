import numpy as np
from numpy.linalg import inv

from LR.process.gradient_descent import gradient_descent


def newton_method(phi, t):
    # init
    w = np.random.rand(3, 1)
    new_w = None
    MAX_Iter = 10000
    eps = 0.001
    learning_rate = 0.001

    # create Hessian (3*3) -> create D
    # Hessian (3, 3) = phi.T(3, 2N) @ D(2N, 2N) @ phi(2N, 3)
    D = np.zeros((phi.shape[0], phi.shape[0]))
    for i in range(phi.shape[0]):
        D[i, i] = np.exp(-phi[i] @ w) / np.power(1 + np.exp(-phi[i] @ w), 2)
    Hessian = phi.T @ D @ phi

    if np.linalg.det(Hessian) == 0:
        # not invertible
        # to gradient descent
        print("Hessian is not invertible. To gradient_descent")
        w = gradient_descent(phi, t)
        return w
    else:
        Hessian_inv = inv(Hessian)

    for now_iter in range(MAX_Iter):
        newton = Hessian_inv @ phi.T @ (1 / (1 + np.exp(-phi @ w)) - t)
        new_w = w - learning_rate * newton
        if np.allclose(new_w, w, rtol=eps):
            print("__converge! early stop!__")
            return new_w
        w = new_w
    return w
