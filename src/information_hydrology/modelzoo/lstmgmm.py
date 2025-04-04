import torch
from torch import nn


class LSTMGMM(nn.Module):
    def __init__(self, input_size, hidden_size, num_gaussians, output_dropout):
        super().__init__()
        """
        LSTM-based Gaussian Mixture Model (mixed density network).
        
        Parameters:
        -----------
            input_size : int
                Number of input features
            hidden_size : int
                Number of hidden units in the LSTM
            num_gaussians : int
                Number of Gaussian components in the mixture
            output_dropout : float
                Dropout rate for the output of the LSTM
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_gaussians = num_gaussians

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, num_gaussians)
        self.fc_sigma = nn.Linear(hidden_size, num_gaussians)
        self.seq_w = nn.Sequential(
            nn.Linear(hidden_size, num_gaussians),
            nn.Softmax(dim=1),
        )
        self.dropout = nn.Dropout(output_dropout)

        self._reset_parameters()

    def forward(self, x):
        """
        The forward function defines the forward pass of a neural network module.
        
        Parameters
        ----------
        x : passed through a LSTM layer.
        
        Returns
        -------
        mu : Computes the mean of the connected layer which likely has a distribution.
        sigma : Computes standard deviation by applying the exponential function to the output of the fully connected layer.
        w : Computes weight, representing additional output from the LSTM's final hidden state.
        """
        _, (h_n, _) = self.lstm(x)
        out = h_n[-1] # many-to-one
        out = self.dropout(out)
        mu = self.fc_mu(out)
        sigma = torch.exp(self.fc_sigma(out))
        w = self.seq_w(out)
        return mu, sigma, w
    
    def sample(self, x, num_samples):
        """
        Sample function generates random samples from a learned distribution based on input.
        
        Parameters
        ----------
        x : input
        num_samples : int
            The number of random samples to generate for each input.
        
        Returns
        -------
        samples : A tensor containing the final weighted random samples, computed based on the learned distribution.
        """
        with torch.no_grad():
            mu, sigma, w = self.forward(x)
            samples = [(torch.randn_like(mu) * sigma + mu).unsqueeze(1) for _ in range(num_samples)]
            samples = torch.cat(samples, dim=1)
            samples = (samples * w.unsqueeze(1)).sum(dim=-1)      
        return samples
    
    def _reset_parameters(self):
        self.lstm.bias_hh_l0.data[self.hidden_size:2 * self.hidden_size] = 3.0
