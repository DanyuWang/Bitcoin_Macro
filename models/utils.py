import numpy as np
import pandas as pd
from MA3.ML4Fin.Bitcoin_Macro.preprocess import data_loader


DAILY, MONTHLY = data_loader()


def load_dataset(x_names, if_ret=False):
    if if_ret:
        log_prix = np.log(DAILY['btc_price'])
        y = log_prix - log_prix.shift()
    else:
        y = DAILY['btc_price']

    x = np.log(DAILY[x_names])

    tmp = pd.concat([x, y], ignore_index=True, axis=1)
    tmp = tmp.dropna()
    x = tmp.iloc[:, :len(x_names)]
    y = tmp.iloc[:, -1]

    return x, y


def k_fold_indice(X, test_size=0.1, k_fold=4):
    test_len = int(len(X) * test_size)
    train_indices = [len(X) - i*test_len for i in range(1, 1+k_fold)]

    return test_len, train_indices


def RRMSE(prediction, target):
    tmp = np.sum(np.power(1-prediction/target, 2))

    return np.sqrt(tmp/len(prediction))

