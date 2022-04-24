import numpy as np


def print_image(p):
    # p:(10, 784)
    threshold = 0.4
    num_group = p.shape[0]
    num_row_pixels = int(np.sqrt(p.shape[1]))
    num_col_pixels = int(np.sqrt(p.shape[1]))
    for now_group in range(num_group):
        print(f"class 第{now_group}群:")
        for now_row_pixel in range(num_row_pixels):
            for now_col_pixel in range(num_col_pixels):
                next_print_bin = 1 if p[now_group, now_row_pixel * num_row_pixels + now_col_pixel] >= threshold else 0
                print(next_print_bin, end=" ")
            print()
        print()


def print_label_image(p, converter):
    # p:(10, 784)
    # converter <index:label, value:group>
    threshold = 0.4
    num_group = p.shape[0]
    num_row_pixels = int(np.sqrt(p.shape[1]))
    num_col_pixels = int(np.sqrt(p.shape[1]))

    for now_focus_label in range(10):
        print(f"labeled class {now_focus_label}:")
        group = int(converter[now_focus_label])  # convert label id  to group id
        for now_row_pixel in range(num_row_pixels):
            for now_col_pixel in range(num_col_pixels):
                next_print_bin = 1 if p[group, now_row_pixel * num_row_pixels + now_col_pixel] >= threshold else 0
                print(next_print_bin, end=" ")
            print()
        print()


def count_confusion(y_train, predict_group, converter):
    # y_train: (60000, 1)
    # predict_group: (60000,) <value: group id>
    # converter: (10,) <index: label, value: group id>
    for now_focus_label in range(10):
        print(f"Confusion Matrix {now_focus_label}:")
        TP = 0
        FN = 0
        FP = 0
        TN = 0
        for now_focus_pic_id in range(60000):
            now_pic_predict_group_id = predict_group[now_focus_pic_id]
            now_pic_predict_label = np.where(converter == now_pic_predict_group_id)[0][0]
            now_pic_real_label = y_train[now_focus_pic_id]
            if now_pic_real_label == now_focus_label and now_pic_predict_label == now_focus_label:
                # TP
                TP += 1
            elif now_pic_real_label != now_focus_label and now_pic_predict_label != now_focus_label:
                # TN
                TN += 1
            elif now_pic_real_label != now_focus_label and now_pic_predict_label == now_focus_label:
                # FP
                FP += 1
            else:
                # FN
                FN += 1

        print(f"\t \t \t \t Predict number {now_focus_label} \t Predict not number {now_focus_label}")
        print(f"Is number {now_focus_label} \t {TP} \t \t \t \t {FN}")
        print(f"Isn't number {now_focus_label} \t {FP} \t \t \t \t{TN}")
        print()
        print(f"Sensitivity (Successfully predict number {now_focus_label})  : {(TP/(TP+FN)):.5f}")
        print(f"Specificity(Successfully predict not number {now_focus_label})  : {(TN / (TN + FP)):.5f}")
        print("-----------------------------------------------------------------------------")


def error_rate(y_train, predict_group, converter):
    count = 0
    for now_focus_pic_id in range(60000):
        now_pic_predict_group_id = predict_group[now_focus_pic_id]
        now_pic_predict_label = np.where(converter == now_pic_predict_group_id)[0][0]
        now_pic_real_label = y_train[now_focus_pic_id]
        if now_pic_predict_label != now_pic_real_label:
            count += 1
    print(f"Total error rate: {count / 60000}")
