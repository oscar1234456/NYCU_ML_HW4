import numpy as np


def count_diff(new_Lamb, Lamb, new_p, p):
    # new_Lamb, Lamb = (10,)
    # new_p, p = (10, 784)
    diff_Lamb = abs(np.sum(new_Lamb - Lamb))
    diff_p = abs(np.sum(new_p - p))

    return diff_Lamb + diff_p

