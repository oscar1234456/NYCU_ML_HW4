import numpy as np


def convert_to_bin(X_train):
    return np.array(X_train >= 128, dtype='uint8')
