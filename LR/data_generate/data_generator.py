import math
import random


import numpy as np
import matplotlib.pyplot as plt
from numpy import log as ln


# m: mean
# s: variance
def random_data_generator(m, s):
    # ref:https://medium.com/mti-technology/how-to-generate-gaussian-samples-3951f2203ab0
    # ref:https://blog.csdn.net/fengdu78/article/details/118715198
    # Box-Muller
    U = np.random.rand()
    V = np.random.rand()
    X = np.sqrt((-2 * ln(U))) * np.cos(2 * np.pi * V)
    # X is drawn from normal distribution N(0, 1)
    # E[ax+b] = a * E[x]+b ; var[ax+b] = a^2 * var[x]
    # Therefore, we need to let a^2 be s => a = sqrt(s)
    a = np.sqrt(s)
    convert_X = a * X + m
    return convert_X
    # U = random.random()
    # V = random.random()
    # z = math.sqrt(-2 * math.log(U)) * math.cos(2 * math.pi * V)
    # sample = z * s + m
    # return sample


def data_generator(N, mx, my, vx, vy):
    generated_data = np.zeros((N, 2))  # 2 dim multivariate Gaussian
    for i in range(N):
        x = random_data_generator(mx, vx)
        y = random_data_generator(my, vy)
        generated_data[i, 0] = x
        generated_data[i, 1] = y
    return generated_data




if __name__ == "__main__":
    samples = list()
    for i in range(20000):
         samples.append(random_data_generator(20,10))
    plt.hist(samples, 50)
    plt.show()
    # x, y = poly_data_generator(2, 10, np.array([2, 5]))
    # print(f"x:{x}, y:{y}")
