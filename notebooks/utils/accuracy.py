import torch
from sklearn.metrics import accuracy_score
import numpy as np


def calculate_accuracy(y_pred, y_true):
    """Function used to compute accuracy for two pytorch tensors"""
    y_true = y_true.cpu().detach().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    y_pred = y_pred.cpu().detach().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_acc = accuracy_score(y_true, y_pred)

    return score_acc


if __name__ == '__main__':
    y_1 = torch.Tensor([1, 2, 3])
    y_2 = torch.Tensor([1, 2, 4])
    print(calculate_accuracy(y_1, y_2))
