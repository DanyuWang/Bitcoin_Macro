import torch
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
    input_dim = 2
    output_dim = 1
    learning_rate = 0.001
    epochs = 500

    model = LinearRegression(input_dim, output_dim)
    # criterion = torch.nn.MSELoss()
    criterion = RMSE()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        inputs = Variable(torch.from_numpy(x_train.values)).float()
        labels = Variable(torch.from_numpy(y_train.values)).float()
        labels = labels.reshape(len(labels), 1)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        if epoch % 50 == 0:
            print('epoch {}, loss {}'.format(epoch, loss.item()))


def loop_k_fold(X, Y, k_fold=4):
    test_len, train_indices = k_fold_indice(len(X), test_size=0.1, k_fold=k_fold)
    for i in range(k_fold):
        print("Training", i+1, "fold")
        x_train = X.iloc[:train_indices[i], :]
        y_train = Y.iloc[:train_indices[i]]
        x_test = X.iloc[train_indices[i]:train_indices[i]+test_len, :]
        y_test = Y.iloc[train_indices[i]:train_indices[i]+test_len]

        train_linearRegression(x_train, y_train)


if __name__ == '__main__':
    X, Y = load_dataset(['flow_in', 'flow_out'], if_ret=True)
    # test_len, train_indices = k_fold_indice(len(X), test_size=0.1, k_fold=4)
    # x_train = X.iloc[:train_indices[0], :]
    # y_train = Y.iloc[:train_indices[0]]
    #
    # train_linearRegression(x_train, y_train)
    loop_k_fold(X, Y)