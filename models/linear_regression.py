import matplotlib.pyplot as plt
from MA3.ML4Fin.Bitcoin_Macro.models.utils import *
from torch.autograd import Variable


class LinearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)

        return out


def train_linearRegression(x_train, y_train):
    input_dim = x_train.shape[1]
    output_dim = 1
    learning_rate = 0.001
    epochs = 1500

    model = LinearRegression(input_dim, output_dim)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_record = []
    inputs = Variable(torch.from_numpy(x_train.values)).float()
    labels = Variable(torch.from_numpy(y_train.values)).float()
    labels = labels.reshape(len(labels), 1)

    for epoch in range(epochs):

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        train_loss_record.append(loss.item())
        if epoch % 50 == 0:
            print('epoch {}, loss {}'.format(epoch, loss.item()))

    return train_loss_record, model


def loop_k_fold(X, Y, k_fold=4):
    loss_k = []
    loss_test_k = []
    criterion = nn.HuberLoss()
    test_len, train_indices = k_fold_indice(len(X), test_size=0.1, k_fold=k_fold)
    for i in reversed(range(k_fold)):
        print("Training", k_fold - i, "fold")
        x_train = X.iloc[:train_indices[i], :]
        y_train = Y.iloc[:train_indices[i]]
        x_test = X.iloc[train_indices[i]:train_indices[i]+test_len, :]
        y_test = Y.iloc[train_indices[i]:train_indices[i]+test_len]

        loss_record, model = train_linearRegression(x_train, y_train, x_test, y_test)
        loss_k.append(loss_record[-1])
        test_labels = Variable(torch.from_numpy(y_test.values)).float()
        test_labels = test_labels.reshape(len(test_labels), 1)
        test_predict = model(Variable(torch.from_numpy(x_test.values)).float())
        loss_test_k.append(criterion(test_predict, test_labels).item())
        test_predict = test_predict.reshape(-1)

    return loss_record, test_predict, y_test, loss_k, loss_test_k


def show_prediction(predict, label, if_cumprod=False):
    predict = pd.Series(predict.detach().numpy(), index=label.index)
    print(np.corrcoef(predict, label))
    if if_cumprod:
        change_true = np.cumprod(1 + label)
        change_pred = np.cumprod(1 + predict)
        plt.plot(change_pred)
        plt.plot(change_true)
        plt.legend(['prediction', 'true'])
        plt.show()
    else:
        plt.plot(label)
        plt.plot(predict)
        plt.legend(['true', 'prediction'])
        plt.show()


if __name__ == '__main__':
    X, Y = load_dataset(['flow_in', 'flow_out'], if_ret=True, shift_step=1, if_cnst=True)
    X['in_2'] = X['flow_in'] * X['flow_in']
    X['out_2'] = X['flow_out'] * X['flow_out']
    # test_len, train_indices = k_fold_indice(len(X), test_size=0.1, k_fold=4)
    # x_train = X.iloc[:train_indices[0], :]
    # y_train = Y.iloc[:train_indices[0]]
    #
    # train_linearRegression(x_train, y_train)
    loss_record, test_predict, y_test, loss_k, loss_k_test = loop_k_fold(X, Y)
    show_prediction(test_predict, y_test, if_cumprod=False)
    print('Average train loss', np.mean(loss_k))
    print('Average test loss', np.mean(loss_k_test))
    plt.plot(loss_record)
    plt.show()
