import numpy as np


def create_group_label_convert(W, y_train):
    # counter: (10, 10) <row: label>
    # converter:(10,) <index: label, value: group>
    predict_group = np.argmax(W, axis=1)  # predict_group: (60000,)
    counter = np.zeros((10, 10))
    converter = np.zeros(10)
    for now_focus_pic_index in range(60000):
        counter[y_train[now_focus_pic_index], predict_group[now_focus_pic_index]] += 1

    for _ in range(10):
        label, group = np.where(counter == np.max(counter))
        label = label[0]
        group = group[0]
        converter[label] = group
        counter[label] = -999
        counter[:, group] = -999

    return predict_group, converter
