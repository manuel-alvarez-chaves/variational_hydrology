from torch import nn


class CudaLSTM(nn.Module):
    def __init__(self, num_inputs, num_hidden):
        super().__init__()
        self.input_size = num_inputs
        self.hidden_size = num_hidden

        self.lstm = nn.LSTM(input_size=num_inputs, hidden_size=num_hidden, batch_first=True)
        self.linear = nn.Linear(num_hidden, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.linear(h_n[-1])