import matplotlib.pyplot as plt
import numpy as np


def pretty_plot(phi, class1_cluster_g, class2_cluster_g, class1_cluster_n, class2_cluster_n, N):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle('Compare pictures')

    # test_datapoint_x_min = test_datapoint_x.min()
    # test_datapoint_x_max = test_datapoint_x.max()
    ax1.set_title("Ground Truth")
    ax1.scatter(phi[:N, 0], phi[:N, 1], c="red")  # plot data-points
    ax1.scatter(phi[N:, 0], phi[N:, 1], c="blue")  # plot data-points

    ax2.set_title("Gradient Descent")
    ax2.scatter(class1_cluster_g[:, 0], class1_cluster_g[:, 1], c="red")  # plot data-points
    ax2.scatter(class2_cluster_g[:, 0], class2_cluster_g[:, 1], c="blue")  # plot data-points

    ax3.set_title("Newton's Method")
    ax3.scatter(class1_cluster_n[:, 0], class1_cluster_n[:, 1], c="red")  # plot data-points
    ax3.scatter(class2_cluster_n[:, 0], class2_cluster_n[:, 1], c="blue")  # plot data-points

    plt.show()
