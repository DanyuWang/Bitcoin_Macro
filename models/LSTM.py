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
    hidden_size = 10
    num_layers = 1
    learning_rate = 0.01
    epochs = 10000 # reduce

    inputs = Variable(torch.from_numpy(x_train.values)).float()
    inputs = inputs.reshape((inputs.shape[0], 1, inputs.shape[1]))
    labels = Variable(torch.from_numpy(y_train.values)).float()
    labels = labels.reshape(len(labels), 1)

    model = LSTM_uni(output_dim, input_dim, hidden_size, num_layers, x_train.shape[1])
    criterion = nn.HuberLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_record = []

    for epoch in range(epochs):
        # test_output = model(test_inputs)
        # test_label
        # loss = criterion (label)
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
    loss_max = np.inf
    test_len, train_indices = k_fold_indice(len(X), test_size=0.1, k_fold=k_fold)
    for i in reversed(range(k_fold)):
        print("Training", i+1, "fold")
        x_train = X.iloc[:train_indices[i], :]
        y_train = Y.iloc[:train_indices[i]]
        x_test = X.iloc[train_indices[i]:train_indices[i]+test_len, :]
        y_test = Y.iloc[train_indices[i]:train_indices[i]+test_len]

        loss_record, model = train_lstm_uni(x_train, y_train)
        loss_k.append(loss_record[-1])
        if loss_k[-1] < loss_max:
            loss_max = loss_k[-1]
            x_all = pd.concat([x_train, x_test])
            y_all = pd.concat([y_train, y_test])
            test_input = Variable(torch.from_numpy(x_all.values)).float()
            test_input = test_input.reshape((test_input.shape[0], 1, test_input.shape[1]))
            test_predict = model(test_input)
            test_predict = test_predict.reshape(-1)
            test_true = y_all

    return loss_record, test_predict, test_true, loss_k


def show_prediction(predict, label):
    predict = pd.Series(predict.detach().numpy(), index=label.index)
    plt.plot(predict)
    plt.plot(label)
    plt.legend(['prediction', 'true'])
    plt.show()


if __name__ == '__main__':
    X, Y = load_dataset(['flow_in', 'flow_out', 'supply', 'ffr', 'gold'], if_ret=False, shift_step=2)
    # test_len, train_indices = k_fold_indice(len(X), test_size=0.1, k_fold=4)
    # x_train = X.iloc[:train_indices[0], :]
    # y_train = Y.iloc[:train_indices[0]]
    #
    # train_lstm_uni(x_train, y_train)
    loss_record, test_predict, y_test, loss_k = loop_k_fold(X, Y)
    show_prediction(test_predict, y_test)
