import numpy as np
from tqdm import tqdm, trange


def get_w_posterior(X_train_bin, Lamb, p):
    # p: (10, 784) , lamb: (10,)
    # return: W (60000, 10) P(z群|圖x,theta) = [P(圖x|z群, theta) * P(Z群|theta)] / P(圖x|theta)
    # bernoulli: (theta ** xi) * ((1-theta) ** (1-xi)), xi == 1 -> pixel bin == 1
    # Assumption: all pixel independent, likelihood function P(圖x|z群, theta) is the product of bernoulli
    # P(Z群|theta) -> lambda
    # TODO: 60000, 10可換成變數

    # Get likelihood P(圖x|z群, theta) = continued product of bernoulli by pixels
    W = np.zeros((60000, 10))

    now_focus_pic_index = trange(0, 60000,  dynamic_ncols=True)

    for now_pic_index in now_focus_pic_index:
        now_focus_pic = X_train_bin[now_pic_index]  # now_focus_pic: [1, 784]
        now_focus_pic_complement = 1 - X_train_bin[now_pic_index]  # X_train_bin is 0/1 matrix
        for j in range(10):
            now_focus_group_prob = p[j]
            now_focus_pic_index.set_description(f"(E)pic{now_pic_index}, c{j}")
            success = now_focus_pic * now_focus_group_prob
            unsuccess = now_focus_pic_complement * (1 - now_focus_group_prob)
            W[now_pic_index, j] = np.prod(success + unsuccess)
    now_focus_pic_index.close()
    # multiple lambda
    W_step2 = W * (Lamb.reshape(1, -1))

    # P(圖x|theta)
    sum_up = np.sum(W_step2, axis=1).reshape(60000, 1)  # sum up all group along pictures. sum_up:(60000, 1)
    np.place(sum_up, sum_up == 0, [1])  # replace 0 to 1

    # run
    W_step3 = W_step2 / sum_up  # normalization

    return W_step3


def get_L(W):
    # Lamb(MLE) = sum(Wi) / n 每group被選到機率
    # Lamb: (10,)
    each_group_sum = np.sum(W, axis=0).reshape(10)
    new_Lamb = each_group_sum / 60000

    return new_Lamb


def get_p(X_train_bin, W):
    # p = sum(wi*xi)/sum(wi)
    # X_train_bin: (60000*784)
    # W: (60000*10)
    sum_up = np.sum(W, axis=0).reshape(10)  # (10)
    np.place(sum_up, sum_up == 0, [1])
    W_normalize = W / sum_up
    p = W_normalize.T @ X_train_bin

    return p
