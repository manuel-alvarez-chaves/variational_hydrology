import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import laplace_asymmetric, norm
from torch import nn

from information_hydrology.utils.distributions import Distribution


class LSTMMDN(nn.Module):
    def __init__(self, input_size, hidden_size, distribution, num_components, output_dropout):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.distribution = distribution
        self.num_components = num_components
        self.output_dropout = output_dropout

        match distribution:
            case Distribution.GAUSSIAN:
                self.num_moments = 2
            case Distribution.LAPLACE:
                self.num_moments = 3

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc_moments = nn.Linear(hidden_size, self.num_moments * num_components)
        self.fc_weights = nn.Sequential(
            nn.Linear(hidden_size, num_components),
            nn.Softmax(dim=1)
        )
        self.dropout = nn.Dropout(output_dropout)

        self._reset_parameters()
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = h_n[-1] # many-to-one
        out = self.dropout(out)
        w = self.fc_weights(out)
        out = self.fc_moments(out)
        match self.distribution:
            case Distribution.GAUSSIAN:
                loc, scale = out.chunk(2, dim=-1)
                scale = F.softplus(scale)
                moments = (loc, scale, None)
            case Distribution.LAPLACE:
                loc, scale, kappa = out.chunk(3, dim=-1)
                scale = F.softplus(scale)
                kappa = F.softplus(kappa)
                moments = (loc, scale, kappa)
        return moments, w
    
    def calc_mean(self, x):
        with torch.no_grad():
            moments, w = self(x)
            match self.distribution:
                case Distribution.GAUSSIAN:
                    loc, scale, _ = moments
                    mean = loc
                case Distribution.LAPLACE:
                    loc, scale, kappa = moments
                    mean = loc + scale * (1 - kappa.pow(2)) / kappa
            mean = (mean * w).sum(axis=1)
        return mean.cpu().numpy()
    
    def sample(self, x, num_samples):
        num_batches = x.shape[0]
        with torch.no_grad():
            moments, w = self(x)
            match self.distribution:
                case Distribution.GAUSSIAN:
                    loc, scale, _ = moments
                    loc, scale = loc.cpu().numpy(), scale.cpu().numpy()
                case Distribution.LAPLACE:
                    loc, scale, kappa = moments
                    loc, scale, kappa = loc.cpu().numpy(), scale.cpu().numpy(), kappa.cpu().numpy()
            w = w.cpu().numpy()

            samples = np.empty((num_batches, num_samples, self.num_components))
            for idx in range(num_batches):
                for idy in range(self.num_components):
                    match self.distribution:
                        case Distribution.GAUSSIAN:
                            samples[idx, :, idy] = norm(loc[idx, idy], scale[idx, idy]).rvs(num_samples)
                        case Distribution.LAPLACE:
                            samples[idx, :, idy] = laplace_asymmetric(kappa=kappa[idx, idy], loc=loc[idx, idy], scale=scale[idx, idy]).rvs(num_samples)
        w =  w[:, np.newaxis, :]
        samples = (samples * w).sum(axis=-1)
        return samples
    
    def _reset_parameters(self):
        self.lstm.bias_hh_l0.data[self.hidden_size:2 * self.hidden_size] = 3.0