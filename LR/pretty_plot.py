import matplotlib.pyplot as plt
import numpy as np


def pretty_plot(test_datapoint_x, test_datapoint_b):
    fig, (ax1, ax2) = plt.subplots(3, 1)
    fig.suptitle('Fitting Curve')
    n = lse_w.shape[0]  # get the degree of the polynomial
    test_datapoint_x_min = test_datapoint_x.min()
    test_datapoint_x_max = test_datapoint_x.max()
    x_space = np.linspace(test_datapoint_x_min-1, test_datapoint_x_max+1) # set x space
    y = np.zeros(len(x_space))
    ax1.scatter(test_datapoint_x, test_datapoint_b, c="red")  # plot data-points

    for i in range(n):
        # y0 = w0x0^0 + w1x0^1 + .... + wn-1x0^n-1
        # y1 = w0x1^0 + w1x1^1 + .... + wn-1x1^n-1
        # x_space = [x0, x1, x2,.....]
        y = y + lse_w[i] * np.power(x_space, i)

    ax1.plot(x_space, y, c="black")  # plot the fitting curve
    ax1.set_title("LSE")
    y = np.zeros(len(x_space))

    for i in range(n):
        y = y + newton_w[i] * np.power(x_space, i)

    ax2.scatter(test_datapoint_x, test_datapoint_b, c="red")
    ax2.plot(x_space, y, c="black")  # plot the fitting curve
    ax2.set_title("Newton's method")
    fig.tight_layout()

    plt.show()