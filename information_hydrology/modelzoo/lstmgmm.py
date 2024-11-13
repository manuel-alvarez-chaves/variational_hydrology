import torch
from torch import nn


class LSTMGMM(nn.Module):
    def __init__(self, input_size, hidden_size, num_gaussians):
        super(LSTMGMM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_gaussians = num_gaussians

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, num_gaussians)
        # self.seq_sigma = nn.Sequential(
        #     nn.Linear(hidden_size, num_gaussians),
        #     nn.Softplus(),
        # )
        self.fc_sigma = nn.Linear(hidden_size, num_gaussians)
        self.seq_w = nn.Sequential(
            nn.Linear(hidden_size, num_gaussians),
            nn.Softmax(dim=1),
        )

        self._reset_parameters()

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = h_n[-1]
        mu = self.fc_mu(out)
        sigma = torch.exp(self.fc_sigma(out))
        w = self.seq_w(out)
        return mu, sigma, w
    
    def sample(self, x, num_samples):
        with torch.no_grad():
            mu, sigma, w = self.forward(x)
            samples = [(torch.randn_like(mu) * sigma + mu).unsqueeze(1) for _ in range(num_samples)]
            samples = torch.cat(samples, dim=1)
            samples = (samples * w.unsqueeze(1)).sum(dim=-1)      
        return samples
    
    def _reset_parameters(self):
        self.lstm.bias_hh_l0.data[self.hidden_size:2 * self.hidden_size] = 3.0
