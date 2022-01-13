import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from MA3.ML4Fin.Bitcoin_Macro.models.utils import *
from sklearn.metrics import mean_absolute_error


def train_randomForest(x_train, y_train, x_test, y_test):
    n_estimator = 1000
    random_state = 32

    rf = RandomForestRegressor(n_estimators=n_estimator, random_state=random_state, criterion='absolute_error')
    rf.fit(x_train, y_train)

    test_prediction = rf.predict(x_test)
    train_prediction = rf.predict(x_train)
    train_loss = mean_absolute_error(y_train, train_prediction)
    test_loss = mean_absolute_error(y_test, test_prediction)

    return test_prediction, train_loss, test_loss


def loop_k_fold(X, Y, k_fold=4):
    loss_k = []
    loss_test_k = []

    test_len, train_indices = k_fold_indice(len(X), test_size=0.1, k_fold=k_fold)
    for i in reversed(range(k_fold)):
        print("Training", k_fold - i, "fold")
        x_train = X.iloc[:train_indices[i], :]
        y_train = Y.iloc[:train_indices[i]]
        x_test = X.iloc[train_indices[i]:train_indices[i]+test_len, :]
        y_test = Y.iloc[train_indices[i]:train_indices[i]+test_len]

        test_prediction, train_loss, test_loss = train_randomForest(x_train, y_train, x_test, y_test)
        loss_k.append(train_loss)
        loss_test_k.append(test_loss)

    return test_prediction, y_test, loss_k, loss_test_k


def show_prediction(predict, label):
    predict = pd.Series(predict, index=label.index)
    plt.plot(label)
    plt.plot(predict)
    plt.legend(['true', 'prediction'])
    plt.show()


if __name__ == '__main__':
    X, Y = load_dataset(['flow_in', 'flow_out'], if_ret=True, shift_step=1, if_cnst=True)
    test_prediction, y_test, loss_k, loss_test_k = loop_k_fold(X, Y)
    show_prediction(test_prediction, y_test)
    print('Average train loss', np.mean(loss_k))
    print('Average test loss', np.mean(loss_test_k))