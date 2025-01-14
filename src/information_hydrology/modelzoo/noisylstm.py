import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal


class NoisyLSTM(nn.Module):
    def __init__(self, num_inputs, num_hidden, output_dropout):
        super().__init__()
        self.input_size = num_inputs
        self.hidden_size = num_hidden

        self.lstm = nn.LSTM(input_size=num_inputs, hidden_size=num_hidden, batch_first=True)
        self.dropout = nn.Dropout(output_dropout)
        self.linear = nn.Linear(num_hidden, 1)

        self.m = MultivariateNormal(torch.zeros(num_hidden), torch.eye(num_hidden))
        self._reset_parameters()

    def forward(self, x, num_samples):
        _, (h_n, _) = self.lstm(x)
        encoded = self.dropout(h_n[-1])
        encoded = encoded.unsqueeze(1).repeat(1, num_samples, 1)
        noise = self.m.sample((x.size(0), num_samples))
        return self.linear(encoded * (1 + noise))

    def _reset_parameters(self):
        self.lstm.bias_hh_l0.data[self.hidden_size:2 * self.hidden_size] = 3.0