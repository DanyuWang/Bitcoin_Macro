import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scipy
from MA3.ML4Fin.Bitcoin_Macro.preprocess import data_loader


DAILY, MONTHLY = data_loader()


def load_dataset(x_names, if_ret=False, shift_step=0, if_cnst=False, add_month=None):
    if if_ret:
        log_prix = np.log(DAILY['btc_price'])
        y = log_prix - log_prix.shift()
    else:
        y = DAILY['btc_price']
    y = y.shift(-shift_step)
    x = DAILY[x_names]
    if add_month is not None:
        month_x = MONTHLY[add_month]
        tmp = pd.concat([x, month_x], axis=1)
        month_x = tmp[add_month]
        month_x = month_x.fillna(method='ffill')
        tmp[add_month] = month_x.iloc[1:, :]
        x = tmp

    tmp = pd.concat([x, y], axis=1)
    tmp = tmp.dropna()

    x = tmp.iloc[:, :-1]
    norm_x = (x - x.mean()) / x.std()
    if if_cnst:
        norm_x['const'] = np.ones(norm_x.shape[0])
    y = tmp.iloc[:, -1]

    return norm_x, y


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