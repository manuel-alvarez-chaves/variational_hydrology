from enum import Enum

import torch
from torch import nn


class SamplingMode(Enum):
    STANDARD = 1
    LEARNED = 2

class VLSTM(nn.Module):
    def __init__(self, num_input, num_hidden):
        super().__init__()
        self.input_size = num_input
        self.hidden_size = num_hidden

        # Encoder | Decoder
        self.encoder = nn.LSTM(input_size=num_input, hidden_size=num_hidden, batch_first=True)
        self.scale = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.Sigmoid(),
        )
        self.decoder = nn.Linear(num_hidden, 1)

        # Variational Inference
        self.mu = nn.Linear(num_hidden, num_hidden)
        self.log_var = nn.Linear(num_hidden, num_hidden)

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std).requires_grad_(False)
        return mu + eps * std
    
    def encode(self, x):
        _, (h_n, _) = self.encoder(x)
        return self.scale(h_n[-1]) # many-to-one
    
    def decode(self, encoded, z):
        return self.decoder(encoded * (1 + z))
    
    def forward(self, x):
        encoded = self.encode(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        z = self.reparametrize(mu, log_var)
        decoded = self.decode(encoded, z)
        return encoded, decoded, mu, log_var
    
    def generate_samples(self, x, num_samples, mode=SamplingMode.STANDARD):
        num_batches = x.shape[0]
        encoded = self.encode(x)
        encoded = encoded.unsqueeze(1).repeat(1, num_samples, 1)
        if mode == SamplingMode.STANDARD:
            z = torch.randn(num_batches, num_samples, self.hidden_size)
        elif mode == SamplingMode.LEARNED:
            mu = self.mu(encoded)
            log_var = self.log_var(encoded)
            z = self.reparametrize(mu, log_var) 
        else:
            raise ValueError("Invalid Sampling Mode")
        return self.decode(encoded, z)

    def sample(self, x, num_samples, mode=SamplingMode.STANDARD, track_grad=False):
        if track_grad:
            return self.generate_samples(x, num_samples, mode)
        else:
            with torch.no_grad():
                return self.generate_samples(x, num_samples, mode)