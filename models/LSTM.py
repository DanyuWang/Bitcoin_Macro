import matplotlib.pyplot as plt
from MA3.ML4Fin.Bitcoin_Macro.models.utils import *
from torch.autograd import Variable


class LSTM_uni(nn.Module):
    def __init__(self, output_size, input_size, hidden_size, num_layers, seq_length):
        super(LSTM_uni, self).__init__()
        self._output_size = output_size
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._seq_len = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.5)
        self.fc_1 = nn.Linear(hidden_size, 256)
        self.fc_2 = nn.Linear(256, output_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self._num_layers, x.size(0), self._hidden_size))
        c_0 = Variable(torch.zeros(self._num_layers, x.size(0), self._hidden_size))

        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self._hidden_size)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_2(out)
        return out


def train_lstm_uni(x_train, y_train):
    input_dim = x_train.shape[1]
    output_dim = 1
    hidden_size = 16
    num_layers = 1
    learning_rate = 0.01
    epochs = 7000

    inputs = Variable(torch.from_numpy(x_train.values)).float()
    inputs = inputs.reshape((inputs.shape[0], 1, inputs.shape[1]))
    labels = Variable(torch.from_numpy(y_train.values)).float()
    labels = labels.reshape(len(labels), 1)

    model = LSTM_uni(output_dim, input_dim, hidden_size, num_layers, x_train.shape[1])
    criterion = nn.HuberLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_record = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        loss_record.append(loss.item())
        if epoch % 200 == 0:
            print('epoch {}, loss {}'.format(epoch, loss.item()))

    return loss_record, model


def loop_k_fold(X, Y, k_fold=4):
    loss_k = []
    loss_k_test = []
    criterion = nn.HuberLoss()
    test_len, train_indices = k_fold_indice(len(X), test_size=0.1, k_fold=k_fold)
    for i in (range(k_fold)):
        print("Training", 1 + i, "fold")
        x_train = X.iloc[:train_indices[i], :]
        y_train = Y.iloc[:train_indices[i]]
        x_test = X.iloc[train_indices[i]:train_indices[i]+test_len, :]
        y_test = Y.iloc[train_indices[i]:train_indices[i]+test_len]

        loss_record, model = train_lstm_uni(x_train, y_train)
        loss_k.append(loss_record[-1])

        x_all = pd.concat([x_train, x_test])
        y_all = pd.concat([y_train, y_test])
        test_input = Variable(torch.from_numpy(x_all.values)).float()
        test_input = test_input.reshape((test_input.shape[0], 1, test_input.shape[1]))
        test_predict = model(test_input)
        labels_test = Variable(torch.from_numpy(y_all.values)).float()
        labels_test = labels_test.reshape(len(labels_test), 1)
        loss_test = criterion(test_predict, labels_test)
        loss_k_test.append(loss_test.item())
        test_predict = test_predict.reshape(-1)

    return loss_record, test_predict, y_all, loss_k, loss_k_test, x_train.index[-1]


def show_prediction(predict, label, line_x):

    predict = pd.Series(predict.detach().numpy(), index=label.index)
    plt.plot(predict)
    plt.plot(label)
    plt.vlines(line_x, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', colors='r')
    plt.legend(['prediction', 'true'])
    plt.show()


if __name__ == '__main__':
    X, Y = load_dataset(['flow_in', 'flow_out', 'ffr', 'supply', 'gold'], if_ret=False, shift_step=22, add_month=['CPI', 'M2'])
    loss_record, test_predict, y_test, loss_k, loss_k_test, line_x = loop_k_fold(X, Y)
    show_prediction(test_predict, y_test, line_x)
    print('Average train loss', np.mean(loss_k))
    print('Average test loss', np.mean(loss_k_test))
    plt.plot(loss_record)
    plt.show()
