import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from MA3.ML4Fin.Bitcoin_Macro.preprocess import data_loader


DAILY, MONTHLY = data_loader()


def load_dataset(x_names, if_ret=False, shift_step=0, non_log=None):
    if if_ret:
        log_prix = np.log(DAILY['btc_price'])
        y = log_prix - log_prix.shift()
    else:
        y = DAILY['btc_price']
    y = y.shift(-shift_step)
    x = np.log(DAILY[x_names])
    if non_log is not None:
        x[non_log] = np.exp(x[non_log])

    tmp = pd.concat([x, y], ignore_index=True, axis=1)
    tmp = tmp.dropna()
    x = tmp.iloc[:, :len(x_names)]
    y = tmp.iloc[:, -1]

    return x, y


def k_fold_indice(len_X, test_size=0.1, k_fold=4):
    test_len = int(len_X * test_size)
    train_indices = [len_X - i*test_len for i in range(1, 1+k_fold)]

    return test_len, train_indices


class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()
        return

    def forward(self, prediction, target):
        RMSE = torch.sqrt(torch.mean(torch.pow(prediction-target, 2)))

        return RMSE