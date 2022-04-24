import pickle

import numpy as np

from EM import config
from EM.dataprocess.convert import convert_to_bin
from EM.dataprocess.dataset import DataSet
import EM.config.constant

from tqdm import tqdm, trange

# Get data
from EM.process.convert import create_group_label_convert
from EM.process.count_error import count_diff
from EM.process.e_step import get_w_posterior, get_L, get_p
from EM.process.imagination import print_image
from EM.process.init_process import init_Lamb, init_p

a = DataSet("./data/")
# X_train, y_train = a.get_training_data()
# X_train:(60000, 784) / y_train: (60000,)

# Temp: Using pickle
with open('./data/X_train.pickle', 'rb') as f:
    X_train = pickle.load(f)
with open('./data/y_train.pickle', 'rb') as f:
    y_train = pickle.load(f)

with open('./data/W.pickle', 'rb') as f:
    W = pickle.load(f)

# convert data set to bin
X_train_bin = convert_to_bin(X_train)

# Init
# Lamb (10,) -> lambda (chance that the group will be picked) / total 10 groups
Lamb = init_Lamb()
# p (10, 784) -> under the 10 groups, the probability of pixel bin == 1 / total 784 pixels
p = init_p()

# total_iter = trange(0, config.constant.MAX_Iter, dynamic_ncols=True)

for now_iter in range(config.constant.MAX_Iter):
    # break, if diff is enough small

    # TODO: remove pickle W
    # Get W
    # W = get_w_posterior(X_train_bin, Lamb, p)

    # Get Lamb
    # total_iter.set_description("get W")
    new_Lamb = get_L(W)
    # Get p
    # total_iter.set_description("get p")
    new_p = get_p(X_train_bin, W)
    # print imagination
    print_image(new_p)
    # diff
    diff = count_diff(new_Lamb, Lamb, new_p, p)
    print(f"No. of Iteration: {now_iter+1}, Difference: {diff}")
    Lamb = new_Lamb
    p = new_p

# Final W
# W = get_w_posterior(X_train_bin, Lamb, p)

# TODO: Count correct
# TODO: matching make convert vector
predict_group, converter = create_group_label_convert(W, y_train)

# TODO: confusion matrix


print("123")
