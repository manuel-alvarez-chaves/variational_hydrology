from enum import Enum

import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal


class SamplingMode(Enum):
    STANDARD = "standard"
    LEARNED = "learned"

class ErrorMode(Enum):
    PROPORTIONAL = "proportional"
    EXPONENTIAL = "exponential"
    DENSE = "dense"

class VLSTM(nn.Module):
    def __init__(self,
                 num_input: int,
                 num_hidden: int,
                 output_dropout: float,
                 error: ErrorMode = ErrorMode.PROPORTIONAL,            
    ):
        super().__init__()
        """
        Variational LSTM (VLSTM) model.

        Parameters:
        -----------
            num_input : int
                Number of input features
            num_hidden : int
                Number of hidden units in the LSTM
            output_dropout : float
                Dropout rate for the output of the encoder
            error : ErrorMode
                Error mode for the VLSTM model (see 'decode')
        """
        self.input_size = num_input
        self.hidden_size = num_hidden
        self.error = error
        self.output_dropout = output_dropout

        # Encoder
        self.encoder = nn.LSTM(input_size=num_input, hidden_size=num_hidden, batch_first=True)
        self.dropout = nn.Dropout(output_dropout)
        
        # Decoder
        self.decoder = nn.Linear(num_hidden, 1)
        self.eps = nn.Sequential(nn.Linear(num_hidden, num_hidden), nn.ReLU())
        self._reset_parameters()

        # Variational Inference
        self.fc_mu = nn.Linear(num_hidden, num_hidden)
        self.fc_log_var = nn.Linear(num_hidden, num_hidden)
        self.m = MultivariateNormal(torch.zeros(num_hidden), torch.eye(num_hidden))

        # Dense layer for Error Mode
        if self.error == ErrorMode.DENSE:
            self.dense = nn.Sequential(nn.Linear(num_hidden, num_hidden), nn.ReLU())

    def _reset_parameters(self):
        self.encoder.bias_hh_l0.data[self.hidden_size:2 * self.hidden_size] = 3.0
    
    def _reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std).requires_grad_(False)
        return mu + eps * std
    
    def encode(self, x):
        _, (h_n, _) = self.encoder(x)
        return self.dropout(h_n[-1]) # many-to-one
    
    def decode(self, encoded, z):
        match self.error:
            case ErrorMode.PROPORTIONAL:
                return self.decoder(encoded * (1 + z))
            case ErrorMode.EXPONENTIAL:
                return self.decoder(encoded * (1 + z.exp()))
            case ErrorMode.DENSE:
                return self.decoder(encoded * (1 + self.dense(z)))
    
    def forward(self, x):
        encoded = self.encode(x)
        mu = self.fc_mu(encoded)
        log_var = self.fc_log_var(encoded)
        z = self._reparametrize(mu, log_var)
        decoded = self.decode(encoded, z)
        return encoded, decoded, mu, log_var
    
    def generate_samples(self, x, num_samples, mode=SamplingMode.STANDARD):
        num_batches = x.shape[0]
        encoded = self.encode(x)
        encoded = encoded.unsqueeze(1).repeat(1, num_samples, 1)
        if mode == SamplingMode.STANDARD:
            z = self.m.sample((num_batches, num_samples)).to(x.device)
        elif mode == SamplingMode.LEARNED:
            mu = self.fc_mu(encoded)
            log_var = self.fc_log_var(encoded)
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
