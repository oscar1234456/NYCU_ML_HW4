import numpy as np


def create_phi(class_1, class_2):
    # class_1: (N, 2), class_2: (N, 2)
    # concat to (2N, 3)
    temp_class_1 = np.append(class_1, np.ones((class_1.shape[0], 1)), axis=1)
    temp_class_2 = np.append(class_2, np.ones((class_2.shape[0], 1)), axis=1)
    return np.concatenate((temp_class_1, temp_class_2))


def create_t(N):
    return np.concatenate((np.zeros((N, 1)), np.ones((N, 1))))
