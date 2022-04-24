import numpy as np


def print_image(p):
    # p:(10, 784)
    threshold = 0.5
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


def count_confusion():
    pass
