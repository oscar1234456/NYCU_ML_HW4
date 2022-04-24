import numpy as np


def confusion_matrix(phi, w, t):
    # phi: (2N, 3)
    # w: (3, 1)
    predict = phi @ w  # predict:(2N, 1)
    np.place(predict, predict <= 0, [0])
    np.place(predict, predict > 0, [1])

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(phi.shape[0]):
        ground_truth = t[i]
        predict_cluster = predict[i]
        if ground_truth == 0 :
            if predict_cluster == 0:
                TP += 1
            else:
                FN += 1
        else:
            if predict_cluster == 1:
                TN += 1
            else:
                FP += 1
    print(f"\t \t \t \t Predict cluster 1 \t Predict cluster 2")
    print(f"Is cluster 1  \t {TP} \t \t \t \t {FN}")
    print(f"Is cluster 2  \t {FP} \t \t \t \t{TN}")
    print()
    print(f"Sensitivity (Successfully predict cluster 1 )  : {(TP / (TP + FN)):.5f}")
    print(f"Specificity(Successfully predict cluster 2)  : {(TN / (TN + FP)):.5f}")
    print("-----------------------------------------------------------------------------")


