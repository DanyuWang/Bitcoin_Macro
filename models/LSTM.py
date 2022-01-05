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

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc_2 = nn.Linear(128, output_size)

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
    input_dim = 5
    output_dim = 1
    learning_rate = 0.001
    epochs = 500